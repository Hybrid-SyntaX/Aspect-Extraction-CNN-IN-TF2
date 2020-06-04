import tensorflow as tf
import tensorflow.keras as K

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Embedding,Reshape #Conv1D

from AspectModelHelperClasses import AspectModelBase, KAspectModel


# from __future__ import annotations


class ConvAspectModel(AspectModelBase):
    # %% Creating model
    def __init__(self,embedding_matrix,num_tokens : int,max_sentence_length,num_tags=4, *args, **kwargs):
        super(ConvAspectModel, self).__init__()
        self.sent_len=None
        self.global_trans_params=None
        self.embedding_matrix=embedding_matrix
        self.max_sentence_length=max_sentence_length
        self.WINDOW_LEN = 3 # code: 3, paper: 2
        self.DIM = 300
        self.DROPOUT_RATE = 0.5
        self.num_tokens=num_tokens#len(word_index)+100
        self.embedding_dim=self.embedding_matrix.shape[1]#dataset_util.getEmbeddingDim(embeddings_index)
        self.stride = 1
        self.DROPOUT_CONV = 0.6
        self.FILTER_SIZE = [1, 2, 3]
        self.NUMBER_OF_FEATURE_MAPS = [300, 300, 300]
        self.NUM_TAGS=num_tags
        self.lr=0.001

    def createEmbeddingLayer(self,int_sequences_input,embedding_dim,num_tokens):

        # START OF EMBEDDING
        embedding_sequence = Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=K.initializers.Constant(self.embedding_matrix),
            trainable=False,
            mask_zero=True
        )(int_sequences_input) # canon
        #Tensor("embedding/Identity:0", shape=(None, 100, 300), dtype=float32)
        #embedding_sequence= tf.nn.embedding_lookup(self.embedding_matrix, int_sequences_input, name = "word_embeddings")
        embedding_sequence = Dropout(self.DROPOUT_RATE)(embedding_sequence)

        squeezed_embedding_sequence = tf.keras.backend.squeeze(tf.image.extract_patches(
            embedding_sequence[:, :, :, tf.newaxis], 
            sizes=[1,self.WINDOW_LEN,self.DIM, 1],strides= [1, self.stride, self.DIM, 1],
            rates=[1, 1, 1, 1], padding="SAME", name=None
        ),axis=2) # 2 -> (None, 100, 900) ,  0 -> (100, 1, 900),
        #patches = K.backend.reshape(squeezed_embedding_sequence,(-1,self.max_sentence_length,self.WINDOW_LEN,self.DIM))
        #Tensor("Reshape:0", shape=(None, 100, 3, 300), dtype=float32)

        patches = Reshape((self.max_sentence_length,self.WINDOW_LEN,self.DIM))(squeezed_embedding_sequence)
        #Tensor("reshape_1/Identity:0", shape = (None, 100, 3, 300), dtype = float32)

        patches_reshaped = K.backend.reshape(patches, (-1, self.WINDOW_LEN, self.DIM))[:, :, :, tf.newaxis]
        #Tensor("strided_slice_1:0", shape=(None, 3, 300, 1), dtype=float32)

        #patches_reshaped= Reshape((self.WINDOW_LEN, self.DIM))(patches)[:, :, :, tf.newaxis]
        ##print('patches: ',patches_reshaped)
        input()

        return patches_reshaped

    def createConv2dLayers(self,patches_reshaped):
        convolution_layers_2d = []

        for i, filter_size in enumerate(self.FILTER_SIZE):

            conv2d = Conv2D(self.NUMBER_OF_FEATURE_MAPS[i],(self.FILTER_SIZE[i],self.DIM)  ,activation="relu",
                        kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.1, seed = None),
                        bias_initializer = tf.constant_initializer(0.1),strides=[1,1],
                        padding='valid')(patches_reshaped)
            #Tensor("conv/Identity:0", shape=(None, 3, 1, 300), dtype=float32) !!

            conv2d_pooled= MaxPool2D(pool_size= ((self.WINDOW_LEN - self.FILTER_SIZE[i] + 1),1 ))(conv2d)
            # Tensor("max_pooling2d/Identity:0", shape=(None, 1, 1, 300), dtype=float32)

            conv2d_pooled_squeezed = K.backend.squeeze(conv2d_pooled,axis=1) # 0 -> (3, 99, 100), 1->(None, 1, 300)
            # Tensor("Squeeze_1:0", shape=(1, 1, 300), dtype=float32)
            conv2d_pooled_squeezed_reshaped = K.backend.reshape(
                conv2d_pooled_squeezed,(-1,self.max_sentence_length,self.NUMBER_OF_FEATURE_MAPS[i]))
            #Tensor("Reshape_2:0", shape=(None, 100, 300), dtype=float32)
            print(conv2d_pooled_squeezed_reshaped)

            convolution_layers_2d.append(conv2d_pooled_squeezed_reshaped)
        return convolution_layers_2d
    def createConv2dLayer(self,patches_reshaped,features_num,filter_size):

        conv2d = Conv2D(features_num,(filter_size,self.DIM)  ,activation="relu",
                    kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.1, seed = None),
                    bias_initializer = tf.constant_initializer(0.1),strides=[1,1],
                    padding='valid')(patches_reshaped)
        #Tensor("conv/Identity:0", shape=(None, 3, 1, 300), dtype=float32) !!

        conv2d_pooled= MaxPool2D(pool_size= ((self.WINDOW_LEN - filter_size + 1),1 ))(conv2d)
        # Tensor("max_pooling2d/Identity:0", shape=(None, 1, 1, 300), dtype=float32)

        conv2d_pooled_squeezed = K.backend.squeeze(conv2d_pooled,axis=1) # 0 -> (3, 99, 100), 1->(None, 1, 300)
        # Tensor("Squeeze_1:0", shape=(1, 1, 300), dtype=float32)
        conv2d_pooled_squeezed_reshaped = K.backend.reshape(
            conv2d_pooled_squeezed,(-1,self.max_sentence_length,features_num))
        #Tensor("Reshape_2:0", shape=(None, 100, 300), dtype=float32)
        print(conv2d_pooled_squeezed_reshaped)

        return conv2d_pooled_squeezed_reshaped
    def createDenseLayers(self,conv2d_layers,size):
        dense_input = K.backend.reshape(conv2d_layers, (-1, size))
        #Tensor("Reshape_5:0", shape=(None, 900), dtype=float32)

        dense = Dense(300, activation="relu",#correct
                            kernel_initializer = K.initializers.GlorotUniform(seed=1227),
                            bias_initializer = tf.zeros_initializer(), #'zeros',
                            kernel_regularizer = K.regularizers.l2(self.lr))(dense_input)
        #Tensor("dense/Identity:0", shape=(None, 300), dtype=float32)
        dense= Dropout(self.DROPOUT_CONV)(dense)


        output = Dense(self.NUM_TAGS, activation=None,
                    kernel_initializer = K.initializers.GlorotUniform(seed=1227),
                    kernel_regularizer = K.regularizers.l2(self.lr),
                    bias_initializer = tf.zeros_initializer())(dense)
        #Tensor("dense_1/Identity:0", shape=(None, 4), dtype=float32)

        #input('go')
        output= K.backend.reshape(output,(-1,self.max_sentence_length,self.NUM_TAGS))
        #Tensor("Reshape_6:0", shape=(None, 100, 4), dtype=float32)

        return output

    # def UNUSED_createConv1DLayer(self,conv2d_layers,size):
    #     #filter_shape = [self.config.conv2_filter_size, size, self.config.conv2_dim]
    #     conv1d = Conv1D(size, self.CONV2_FILTER_SIZE, activation="relu", strides = 1, padding = "same",
    #                         kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.1, seed = None),
    #                         bias_initializer = tf.constant_initializer(0.1),)(conv2d_layers)
    #
    #     return conv1d
        #Tensor("conv1d/Identity:0", shape=(None, 100, 900), dtype=float32)
    def createKerasModel(self) :

        int_sequences_input = K.Input(shape=(self.max_sentence_length,), dtype="int64")
        patches_reshaped = self.createEmbeddingLayer(int_sequences_input,self.embedding_dim,self.num_tokens)

        conv2d_layers=K.layers.concatenate([
            self.createConv2dLayer(patches_reshaped,300,1),
            self.createConv2dLayer(patches_reshaped, 300, 2),
            self.createConv2dLayer(patches_reshaped, 300, 3)],axis=2)

        #convolution_layers_2d=self.createConv2dLayers(patches_reshaped)
        #conv2d_layers = K.layers.concatenate(convolution_layers_2d,axis=2)

        #Tensor("concatenate/Identity:0", shape=(None, 100, 900), dtype=float32) ????

        # size = 0
        # for i in range(len(self.FILTER_SIZE)):
        #     size += self.NUMBER_OF_FEATURE_MAPS[i]
        # print('size: ',size)

        size = 0
        for i in range(len(self.FILTER_SIZE)):
            size += self.NUMBER_OF_FEATURE_MAPS[i]
        print('size: ', size)


        output=self.createDenseLayers(conv2d_layers,size)

        return KAspectModel(inputs=int_sequences_input,outputs= output)


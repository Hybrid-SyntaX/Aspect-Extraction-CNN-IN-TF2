import tensorflow as tf
import tensorflow.keras as K

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Embedding,Reshape,Lambda#Conv1D
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, LSTM, Activation

from AspectModelHelperClasses import AspectModelBase, KAspectModel


# from __future__ import annotations


class POSAspectModel(AspectModelBase):
    # %% Creating model
    def __init__(self,num_tokens : int,max_sentence_length,num_tags=4, *args, **kwargs):
        super(POSAspectModel, self).__init__()
        self.embedding_matrix=None#embedding_matrix
        self.max_sentence_length=max_sentence_length
        self.WINDOW_LEN = 3 # code: 3, paper: 2
        self.DIM = 6
        self.DROPOUT_RATE = 0.5
        self.num_tokens=num_tokens#len(word_index)+100
        #self.embedding_dim=self.embedding_matrix.shape[1]#dataset_util.getEmbeddingDim(embeddings_index)
        self.stride = 1
        self.DROPOUT_CONV = 0.6
        self.conv2d_filter_sizes = [3, 2, 1]
        self.conv2d_feature_maps = [300, 100, 50]
        self.num_pos_tags=6
        #self.conv2d_filter_sizes = [1,2,3]
        #self.conv2d_feature_maps = [300, 300, 300]

        #self.FILTER_SIZE = [1, 2, 3]
        #self.NUMBER_OF_FEATURE_MAPS = [300, 300, 300]
        self.NUM_TAGS=num_tags
        self.lr=0.001
    def toDict(self):
        return {
            'model_class': type(self).__name__,
            'learning_rate': self.lr,
            'dropout_rate': self.DROPOUT_RATE,
            'conv_droout_rate': self.DROPOUT_CONV,
            'filter_sizes': self.conv2d_filter_sizes,
            'feature_maps_count': self.conv2d_feature_maps,
            'classes_count': self.NUM_TAGS,
            'window_length': self.WINDOW_LEN,
            'dim': self.DIM,
            #'embedding_dim': self.embedding_dim,
            'word_counts': self.num_tokens,
            'max_sentence_length': self.max_sentence_length
        }
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


        return patches_reshaped
    def createConv2dLayersTF(self,patches_reshaped):
        convolution_layers_2d = []

        #for i, filter_size in enumerate(self.FILTER_SIZE):
        for feature_map, filter_size in zip(self.conv2d_feature_maps, self.conv2d_filter_sizes):
            filter_shape = [filter_size, self.DIM, 1, feature_map]

            W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev = 0.1))
            b = tf.Variable(tf.constant(0.1, shape = [feature_map]))


            conv = tf.nn.conv2d(patches_reshaped, filters = W, strides = [1, 1, 1, 1], padding = "VALID")

            conv2d_pooled = tf.nn.max_pool(conv, ksize = [1, (self.WINDOW_LEN - filter_size + 1), 1, 1],
                                    strides = [1, 1, 1, 1], padding = 'VALID', data_format = 'NHWC', name = "pool")

            #conv2d_pooled_squeezed= Lambda(lambda x: K.backend.squeeze(x,axis=1))(conv2d_pooled)
            conv2d_pooled_squeezed = K.backend.squeeze(conv2d_pooled, axis = 1)  # 0 -> (3, 99, 100), 1->(None, 1, 300)
            #print(conv2d_pooled_squeezed)
            #input()
            # Tensor("Squeeze_1:0", shape=(None, 1, 300), dtype=float32)

            conv2d_pooled_squeezed_reshaped = K.backend.reshape(
                conv2d_pooled_squeezed,(-1,self.max_sentence_length,feature_map))

            #conv2d_pooled_squeezed_reshaped=Reshape((-1,self.max_sentence_length,self.NUMBER_OF_FEATURE_MAPS[i]))(conv2d_pooled_squeezed)
            #Tensor("Reshape_2:0", shape=(None, 100, 300), dtype=float32)
            #print(conv2d_pooled_squeezed_reshaped)

            convolution_layers_2d.append(conv2d_pooled_squeezed_reshaped)
        return convolution_layers_2d
    def createConv2dLayers(self,patches_reshaped):
        convolution_layers_2d = []

        #for i, filter_size in enumerate(self.FILTER_SIZE):
        for feature_map, filter_size in zip(self.conv2d_feature_maps, self.conv2d_filter_sizes):

            conv2d = Conv2D(filters=feature_map,
                            kernel_size=(filter_size,self.DIM)  ,activation="relu",
                        kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.1, seed = None),
                        bias_initializer = tf.constant_initializer(0.1),strides=[1,1],
                        padding='valid')(patches_reshaped)
            #Tensor("conv/Identity:0", shape=(None, 3, 1, 300), dtype=float32) !!

            #conv2d_pooled= MaxPool2D(pool_size= ((self.WINDOW_LEN - filter_size + 1),1),padding='valid',data_format='channels_last')(conv2d)
            conv2d_pooled = MaxPool2D(pool_size = ((self.WINDOW_LEN - filter_size + 1), 1), padding = 'valid',
                                      data_format = 'channels_last')(conv2d)
            # conv2d_pooled = MaxPool2D(pool_size = (2, 1), padding = 'valid',
            #                           data_format = 'channels_last')(conv2d)

            #input()
            #conv2d_pooled = MaxPool2D(pool_size = (1, 1))(conv2d)
            # Tensor("max_pooling2d/Identity:0", shape=(None, 1, 1, 300), dtype=float32)

            conv2d_pooled_squeezed = K.backend.squeeze(conv2d_pooled,axis=1) # 0 -> (3, 99, 100), 1->(None, 1, 300)
            #conv2d_pooled_squeezed= Lambda(lambda x: K.backend.squeeze(x,axis=1))(conv2d_pooled)
            #print(conv2d_pooled_squeezed)
            #input()
            # Tensor("Squeeze_1:0", shape=(None, 1, 300), dtype=float32)
            conv2d_pooled_squeezed_reshaped = K.backend.reshape(
                conv2d_pooled_squeezed,(-1,self.max_sentence_length,feature_map))

            #conv2d_pooled_squeezed_reshaped=Reshape((-1,self.max_sentence_length,self.NUMBER_OF_FEATURE_MAPS[i]))(conv2d_pooled_squeezed)
            #Tensor("Reshape_2:0", shape=(None, 100, 300), dtype=float32)
            print(conv2d_pooled_squeezed_reshaped)

            convolution_layers_2d.append(conv2d_pooled_squeezed_reshaped)
        return convolution_layers_2d
    def createConv2dLayer(self,patches_reshaped,features_num,filter_size,pool_size):

        conv2d = Conv2D(features_num,(filter_size,self.DIM)  ,activation="relu",
                    kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.1, seed = None),
                    bias_initializer = tf.constant_initializer(0.1),strides=[1,1],
                    padding='valid')(patches_reshaped)
        #Tensor("conv/Identity:0", shape=(None, 3, 1, 300), dtype=float32) !!

        conv2d_pooled= MaxPool2D(pool_size= ((self.WINDOW_LEN - filter_size + 1),pool_size ))(conv2d)
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

        int_sequences_input = K.Input(shape=(self.max_sentence_length,self.num_pos_tags))

        # squeezed_embedding_sequence = tf.keras.backend.squeeze(tf.image.extract_patches(
        #     int_sequences_input[:, :, :, tf.newaxis],
        #     sizes=[1,self.WINDOW_LEN,self.DIM, 1],strides= [1, self.stride, self.DIM, 1],
        #     rates=[1, 1, 1, 1], padding="SAME", name=None
        # ),axis=2)
        #
        # patches = Reshape((self.max_sentence_length,self.WINDOW_LEN,self.DIM))(squeezed_embedding_sequence)
        # #Tensor("reshape_1/Identity:0", shape = (None, 100, 3, 300), dtype = float32)
        #
        # patches_reshaped = K.backend.reshape(patches, (-1, self.WINDOW_LEN, self.DIM))[:, :, :, tf.newaxis]
        #
        # convolution_layers_2d=self.createConv2dLayers(patches_reshaped)
        # conv2d_layers = K.layers.concatenate(convolution_layers_2d,axis=2)

        #Tensor("concatenate/Identity:0", shape=(None, 100, 900), dtype=float32) ????

        # size = 0
        # for i in range(len(self.FILTER_SIZE)):
        #     size += self.NUMBER_OF_FEATURE_MAPS[i]
        # print('size: ',size)

        size = 0
        for i in range(len(self.conv2d_filter_sizes)):
            size += self.conv2d_feature_maps[i]
        print('size: ', size)

        my_layer = Bidirectional(LSTM(300, return_sequences = True))(int_sequences_input)

        my_layer = TimeDistributed(Dense(3, kernel_initializer = K.initializers.GlorotUniform(seed = 1227),
                                         kernel_regularizer = K.regularizers.l2(0.001),
                                         bias_initializer = tf.zeros_initializer()))(my_layer)



        output = K.backend.reshape(my_layer, (-1, self.max_sentence_length, self.NUM_TAGS))


        #output=self.createDenseLayers(conv2d_layers,size)

        return KAspectModel(inputs=int_sequences_input,outputs= output)


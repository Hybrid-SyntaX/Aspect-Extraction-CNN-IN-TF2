import tensorflow as tf
import tensorflow.keras as K

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Embedding,Reshape,Flatten ,Lambda#Conv1D

from AspectModelHelperClasses import AspectModelBase, KAspectModel


# from __future__ import annotations


class ConvAspectModel(AspectModelBase):
    # %% Creating model
    def __init__(self,embedding_matrix,num_tokens : int,max_sentence_length,num_tags=4, *args, **kwargs):
        super(ConvAspectModel, self).__init__()
        self.embedding_matrix=embedding_matrix
        self.max_sentence_length=max_sentence_length
        self.WINDOW_LEN = 5 # code: 3, paper: 5 @todo: create a model with 5
        self.DIM = 300
        self.num_tokens=num_tokens#len(word_index)+100
        self.embedding_dim=self.embedding_matrix.shape[1]#dataset_util.getEmbeddingDim(embeddings_index)
        self.stride = 1
        self.DROPOUT_RATE=0.5
        self.DROPOUT_CONV = 0.6
        #self.conv2d_filter_sizes = [3, 2, 1]
        self.conv2d_feature_maps = [300, 100, 50]
        self.conv2d_filter_sizes = [1,2,3]
        #self.conv2d_feature_maps = [300, 300, 300]

        #self.FILTER_SIZE = [1, 2, 3]
        #self.NUMBER_OF_FEATURE_MAPS = [300, 300, 300]
        self.NUM_TAGS=num_tags
        self.lr=0.001 #@try: variable learning rate


    def createEmbeddingLayer(self,int_sequences_input,embedding_dim,num_tokens):

        # START OF EMBEDDING
        embedding_sequence = Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=K.initializers.Constant(self.embedding_matrix),
            trainable=False, #default false
            mask_zero=True
        )(int_sequences_input) # canon
        #Tensor("embedding/Identity:0", shape=(None, 100, 300), dtype=float32)
        #embedding_sequence= tf.nn.embedding_lookup(self.embedding_matrix, int_sequences_input, name = "word_embeddings")
        embedding_sequence = Dropout(self.DROPOUT_RATE)(embedding_sequence)

        extracted_patches = tf.image.extract_patches(
            images = embedding_sequence[:, :, :, tf.newaxis],
            sizes=[1,self.WINDOW_LEN,self.DIM, 1],  # 3, 300
            strides= [1, self.stride, self.DIM, 1], # 1, 300
            rates=[1, 1, 1, 1],
            padding="SAME"
        )
        #Tensor("ExtractImagePatches:0", shape = (None, 65, 1, 900), dtype = float32)
        squeezed_extracted_patches = tf.keras.backend.squeeze(extracted_patches,axis=2) # 2 -> (None, 100, 900) ,  0 -> (100, 1, 900),
        #Tensor("Squeeze:0", shape = (None, 65, 900), dtype = float32)

        #patches = K.backend.reshape(squeezed_embedding_sequence,(-1,self.max_sentence_length,self.WINDOW_LEN,self.DIM))
        #Tensor("Reshape:0", shape=(None, 100, 3, 300), dtype=float32)

        patches = Reshape((self.max_sentence_length,self.WINDOW_LEN,self.DIM))(squeezed_extracted_patches)# 64,3,300
        #Tensor("reshape_1/Identity:0", shape = (None, 100, 3, 300), dtype = float32)

        patches_reshaped = K.backend.reshape(patches, (-1, self.WINDOW_LEN, self.DIM))[:, :, :, tf.newaxis]
        #Tensor("strided_slice_1:0", shape=(None, 3, 300, 1), dtype=float32)

        #patches_reshaped= Reshape((self.WINDOW_LEN, self.DIM))(patches)[:, :, :, tf.newaxis]
        ##print('patches: ',patches_reshaped)

        return patches_reshaped

    def createConv2dLayers(self,patches_reshaped):
        convolution_layers_2d = []
        #TODO: DO NOT USE 'same' convolution mode for 1x1 filters
        for feature_map, filter_size in zip(self.conv2d_feature_maps, self.conv2d_filter_sizes):

            conv2d = Conv2D(filters=feature_map,
                        kernel_size=(filter_size,self.DIM)  ,activation="relu",
                        kernel_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.1, seed = None),
                        bias_initializer = tf.constant_initializer(0.1),strides=[1,1],
                        padding='valid')(patches_reshaped)

            #1. Tensor("conv2d/Identity:0", shape=(None, 1, 1, 300), dtype=float32)
            #2. Tensor("conv2d_1/Identity:0", shape=(None, 2, 1, 100), dtype=float32)
            #3. Tensor("conv2d_2/Identity:0", shape=(None, 3, 1, 50), dtype=float32)
            #Tensor("conv/Identity:0", shape=(None, 3, 1, 300), dtype=float32) !!

            #conv2d_pooled= MaxPool2D(pool_size= ((self.WINDOW_LEN - filter_size + 1),1),padding='valid',data_format='channels_last')(conv2d)
            conv2d_pooled = MaxPool2D(pool_size = (self.WINDOW_LEN - filter_size + 1, 1), 
            padding = 'valid',strides=(1,1),
                                      data_format = 'channels_last')(conv2d) #

            #1.  Tensor("max_pooling2d/Identity:0", shape = (None, 1, 1, 300), dtype = float32) #(3, 1) / (1,1)
            #2.  Tensor("max_pooling2d_1/Identity:0", shape = (None, 1, 1, 100), dtype = float32) #(4,1) / (2,1)
            #3.  Tensor("max_pooling2d_2/Identity:0", shape = (None, 1, 1, 50), dtype = float32) #(5,1) / (3,1)
            print(conv2d_pooled)
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
            print(conv2d_pooled_squeezed_reshaped)
            #conv2d_pooled_squeezed_reshaped=Reshape((-1,self.max_sentence_length,self.NUMBER_OF_FEATURE_MAPS[i]))(conv2d_pooled_squeezed)
            #Tensor("Reshape_2:0", shape=(None, 100, 300), dtype=float32)
            #print(conv2d_pooled_squeezed_reshaped)

            convolution_layers_2d.append(conv2d_pooled_squeezed_reshaped)
        return convolution_layers_2d

    def createDenseLayers(self,conv2d_layers,size):
        dense_input = K.backend.reshape(conv2d_layers, (-1, size))
        #dense_input = Reshape((-1,size))(conv2d_layers)
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

    def createKerasModel(self) :

        int_sequences_input = K.Input(shape=(self.max_sentence_length,), dtype="int64")
        patches_reshaped = self.createEmbeddingLayer(int_sequences_input,self.embedding_dim,self.num_tokens)

        convolution_layers_2d=self.createConv2dLayers(patches_reshaped)
        conv2d_layers = K.layers.concatenate(convolution_layers_2d,axis=2)
        #Tensor("concatenate/Identity:0", shape=(None, 100, 900), dtype=float32) ????

        size = 0
        for i in range(len(self.conv2d_filter_sizes)):
            size += self.conv2d_feature_maps[i]
        print('size: ', size)


        output=self.createDenseLayers(conv2d_layers,size)

        return KAspectModel(inputs=int_sequences_input,outputs= output)

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
            'embedding_dim': self.embedding_dim,
            'word_counts': self.num_tokens,
            'max_sentence_length': self.max_sentence_length
        }
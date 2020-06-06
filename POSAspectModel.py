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
        self.max_sentence_length=max_sentence_length
        self.WINDOW_LEN = 3 # code: 3, paper: 2
        self.DIM = 6
        self.num_tokens=num_tokens#len(word_index)+100

        self.stride = 1
        self.num_pos_tags=6

        self.NUM_TAGS=num_tags
        self.lr=0.001
    def toDict(self):
        return {
            'model_class': type(self).__name__,
            'learning_rate': self.lr,
            'classes_count': self.NUM_TAGS,
            'window_length': self.WINDOW_LEN,
            'dim': self.DIM,
            #'embedding_dim': self.embedding_dim,
            'word_counts': self.num_tokens,
            'max_sentence_length': self.max_sentence_length
        }

    def createKerasModel(self) :

        int_sequences_input = K.Input(shape=(self.max_sentence_length,self.num_pos_tags))

        my_layer = Bidirectional(LSTM(300, return_sequences = True))(int_sequences_input)

        my_layer = TimeDistributed(Dense(3, kernel_initializer = K.initializers.GlorotUniform(seed = 1227),
                                         kernel_regularizer = K.regularizers.l2(0.001),
                                         bias_initializer = tf.zeros_initializer()))(my_layer)



        output = K.backend.reshape(my_layer, (-1, self.max_sentence_length, self.NUM_TAGS))


        #output=self.createDenseLayers(conv2d_layers,size)

        return KAspectModel(inputs=int_sequences_input,outputs= output)


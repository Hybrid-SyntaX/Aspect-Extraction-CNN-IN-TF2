import numpy as np
from joblib import dump

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LambdaCallback, TensorBoard
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling1D,MaxPooling1D, Conv1D
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Activation

from AspectModelHelperClasses import AspectModelAuxData, AspectModelMetadata, crf_accuracy, crf_fscore, crf_precision, \
    crf_recall, crf_loglikelihood_loss, KAspectModel
import tensorflow.keras as K
from ConvAspectModel import ConvAspectModel
from DatasetReader import DatasetReader
import dataset_util
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Embedding,Reshape,Lambda#Conv1D

from helper_util import use_cpu
from sklearn.preprocessing import OneHotEncoder
# %% Use cpu
tf.config.set_soft_device_placement(False)
#tf.config.experimental_run_functions_eagerly(True) #reqruied for GPU
use_cpu(True)
#%% Reading data

max_sentence_length=65
#nostopwords="-nostopwords"

# Load data
embeddings_filename=r'data/sentic2vec-utf8.csv'
restaurantDataset= DatasetReader(f'data/Restaurants_Train_v2.xml.iob',
                                 f'data/Restaurants_Test_Data_phaseB.xml.iob',
                                 'data/aspect-tags.txt',
                                 embeddings_filename,
                                 max_sentence_length)
#x_train,y_train,x_val,y_val=restaurantDataset.prepareData()

# Preprocess data
x_train,x_train_pos,y_train , x_val,x_val_pos,y_val= restaurantDataset.prepareDataForPos()


print(np.shape(x_train))
print(np.shape(x_train_pos))
print(np.shape(y_train))
# 65 x 6

lr=0.001
DROPOUT_CONV=0.6
NUM_TAGS=3


int_sequences_input = K.Input(shape=(max_sentence_length,6))#, dtype="int64"

my_layer= Bidirectional(LSTM(300, return_sequences=True))(int_sequences_input)
my_layer= Bidirectional(LSTM(100, return_sequences=True))(my_layer)


my_layer= TimeDistributed(Dense(3,kernel_initializer = K.initializers.GlorotUniform(seed=1227),
            kernel_regularizer = K.regularizers.l2(0.001),
            bias_initializer = tf.zeros_initializer()))(my_layer)
output= Activation('softmax')(my_layer)
model = KAspectModel(int_sequences_input, output)
model.summary()
model.compile(optimizer="adam", metrics=[])#loss='sparse_categorical_crossentropy',

model.fit(x_train_pos, y_train, batch_size=30, epochs=30)
model.save('models/pos_model_3.h5')
dump(AspectModelAuxData().ViterbiTransParams, 'models/pos_model_3' + '-trans_params.joblib')
#print('F1-score:',model.evaluate_with_f1(x_val_pos,y_val))

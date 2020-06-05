import numpy as np
from joblib import dump
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LambdaCallback, TensorBoard
import tensorflow as tf
from AspectModelHelperClasses import AspectModelAuxData, AspectModelMetadata, crf_accuracy, crf_fscore,crf_precision,crf_recall,crf_loglikelihood_loss
import tensorflow.keras as K
from ConvAspectModel import ConvAspectModel
from DatasetReader import DatasetReader
import dataset_util
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
#x_train,x_train_pos,y_train = restaurantDataset.prepareDataForPos(x_train,y_train)
#x_val,x_val_pos,y_val = restaurantDataset.prepareDataForPos(x_val,y_val)
x_train,x_train_pos,y_train , x_val,x_val_pos,y_val= restaurantDataset.prepareDataForPos()


print(np.shape(x_train))
print(np.shape(x_train_pos))
print(np.shape(y_train))




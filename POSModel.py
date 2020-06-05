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

embeddings_filename=r'data/sentic2vec-utf8.csv'
restaurantDataset= DatasetReader(f'data/Restaurants_Train_v2.xml.iob',
                                 f'data/Restaurants_Test_Data_phaseB.xml.iob',
                                 'data/aspect-tags.txt',
                                 embeddings_filename,
                                 max_sentence_length)
x_train,y_train,x_val,y_val=restaurantDataset.prepareData()

x_train,x_train_pos,y_train = restaurantDataset.prepareDataForPos(x_train,y_train)
x_val,x_val_pos,y_val = restaurantDataset.prepareDataForPos(x_val,y_val)



print(x_train[0])
print(x_train_pos[0])
print(y_train[0])

# for s_i in range(len(x_train)-1):
#     #sentence_pos = x_train_pos_onehot[s_i]
#     #sent_len=len(sentence_pos)
#     sent_len=max(len(x_train_pos_onehot[s_i]),
#                  len(x_train),
#                  len(y_train))
#     new_sentence=[]
#     new_sentence_pos=[]
#     new_sentence_label=[]
#     for w_i in range(sent_len-1):
#         if x_train_pos_onehot[s_i][w_i] is not None:
#             new_sentence.append(x_train[s_i][w_i])
#             new_sentence_pos.append(x_train_pos_onehot[s_i][w_i])
#             new_sentence_label.append(y_train[s_i][w_i])
#     new_sentences.append(new_sentence)
#     new_sentence_poses.append(new_sentence_pos)
#     new_sentence_labels.append(new_sentence_label)





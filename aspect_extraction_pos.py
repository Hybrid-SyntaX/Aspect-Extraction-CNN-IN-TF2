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

from POSAspectModel import POSAspectModel
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

posAspectModel=POSAspectModel(
    num_tokens= len(restaurantDataset.word_index)+100,
    max_sentence_length=restaurantDataset.max_sentence_length,
    num_tags = len(restaurantDataset.labels_dict))

model =posAspectModel.createKerasModel()

# model.summary()
# model.compile(optimizer="adam", metrics=[])#loss='sparse_categorical_crossentropy',
#
# model.fit(x_train_pos, y_train, batch_size=30, epochs=90)
# model.save('models/pos_model_5.h5')
# dump(AspectModelAuxData().ViterbiTransParams, 'models/pos_model_5' + '-trans_params.joblib')
# #print('F1-score:',model.evaluate_with_f1(x_val_pos,y_val))
#

posAspectModelMetadata= AspectModelMetadata(posAspectModel)

metric_fns=[crf_fscore,crf_precision,crf_recall]#crf_accuracy,


#metric_fns=[]

optimizer='adam'
epochs=47 #original: 200 , but ran for 47 epochs
batch_size=30#30

model.compile(optimizer=optimizer, metrics=metric_fns)


modelMetadata=posAspectModelMetadata.createModelMetadata(metric_fns, epochs, batch_size, optimizer)

model.summary()
print(modelMetadata)
input('press to beign training!')



callbacks=[#LambdaCallback(on_batch_end  = lambda batch, logs: updateSentenceLengths(batch) ),
            LambdaCallback(on_epoch_end = lambda epoch, logs: posAspectModelMetadata.updateMetadata(logs)),
           ModelCheckpoint(filepath= posAspectModelMetadata.newFilename() + '-low-loss.h5', monitor= 'val_loss', mode= 'min', save_best_only=True, verbose=1, ),
           ModelCheckpoint(filepath= posAspectModelMetadata.newFilename() + '-high-fscore.h5', monitor= 'val_crf_fscore', mode= 'max', save_best_only=True, verbose=1, ),
           LambdaCallback(on_epoch_end = lambda epoch, logs: dump(AspectModelAuxData().ViterbiTransParams,
                                                                  posAspectModelMetadata.newFilename() + '-trans_params.joblib')),
           #EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25),
           #EarlyStopping(monitor='val_crf_fscore', mode='max', verbose=1,patience=25),
           CSVLogger(posAspectModelMetadata.newFilename() + '.log'),
           TensorBoard(log_dir = "models/tensorflow-results", histogram_freq = 0,
                       write_graph = True, write_images = True,
                       update_freq = 'epoch', profile_batch = 2, embeddings_freq = 0),
           LambdaCallback(on_epoch_end = lambda epoch, logs: posAspectModelMetadata.saveMetadata(model))]
#%% Training model

history = model.fit(x_train_pos, y_train, batch_size=batch_size, epochs=epochs,workers=8,
                    use_multiprocessing=True,validation_data=(x_val_pos, y_val),callbacks=callbacks)


#%% Saving model
posAspectModelMetadata.updateMetadata(history.history)
model.save(posAspectModelMetadata.newFilename() + '.h5')
dump(AspectModelAuxData().ViterbiTransParams, posAspectModelMetadata.newFilename() + '-trans_params.joblib')
posAspectModelMetadata.saveMetadata(model)

print("Trainig complete")


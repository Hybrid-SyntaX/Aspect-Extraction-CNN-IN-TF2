# %% Importing libraries
import numpy as np
from joblib import dump
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LambdaCallback, TensorBoard
import tensorflow as tf
from AspectModelHelperClasses import AspectModelAuxData, AspectModelMetadata, crf_accuracy, crf_fscore,crf_precision,crf_recall,crf_loglikelihood_loss

from ConvAspectModel import ConvAspectModel
from DatasetReader import DatasetReader
from helper_util import use_cpu

# %% Use cpu
tf.config.set_soft_device_placement(False)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#input('Press any key to cotinue...')

tf.config.experimental_run_functions_eagerly(True) #reqruied for GPU

#use_cpu(True)
#%% Reading data

max_sentence_length=65
#embeddings_filename=r'P:\Ongoing\University\Data Mining\Aspect Extraction\_\aspect-extraction-master\glove\glove.840B.300d.txt'
embeddings_filename=r'data/sentic2vec-utf8.csv'
restaurantDataset= DatasetReader('data/Restaurants_Train_v2.xml.iob',
                                 'data/Restaurants_Test_Data_phaseB.xml.iob',
                                 'data/aspect-tags.txt',
                                 embeddings_filename,
                                 max_sentence_length)
x_train,y_train,x_val,y_val=restaurantDataset.prepareData()
# 3 class model
#restaurantDataset.labels_dict.pop('NN') # remove NN class

# 2 class model
#y_train[y_train==restaurantDataset.labels_dict['B-A']] = restaurantDataset.labels_dict['I-A']
#y_val[y_val==restaurantDataset.labels_dict['B-A']] = restaurantDataset.labels_dict['I-A']
#print(y_train[:5])
#restaurantDataset.labels_dict.pop('B-A')

#print(restaurantDataset.labels_dict)
#input()
# %% Creating model
convAspectModel=ConvAspectModel(
    embedding_matrix=restaurantDataset.embedding_matrix,
    num_tokens= len(restaurantDataset.word_index)+100,
    max_sentence_length=restaurantDataset.max_sentence_length,
    num_tags = len(restaurantDataset.labels_dict))
model =convAspectModel.createKerasModel()
convAspectModelMetadata= AspectModelMetadata(convAspectModel)

metric_fns=[crf_fscore,crf_precision,crf_recall]#crf_accuracy,
#metric_fns=[]
optimizer='adam'
epochs=10 #original: 200 , but ran for 47 epochs
batch_size=30#30

model.compile(optimizer=optimizer, metrics=metric_fns)


modelMetadata=convAspectModelMetadata.createModelMetadata(metric_fns,epochs,batch_size,optimizer)

model.summary()
print(modelMetadata)
input('press to beign training!')

def updateSentenceLengths(batch):
    print(batch)
    input()
    AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
    print(AspectModelAuxData().SentencesLength [0])
    input()
# %% Create callbacks

callbacks=[#LambdaCallback(on_batch_end  = lambda batch, logs: updateSentenceLengths(batch) ),
            LambdaCallback(on_epoch_end = lambda epoch, logs: convAspectModelMetadata.updateMetadata(logs)),
           ModelCheckpoint(filepath= convAspectModelMetadata.newFilename() + '-low-loss.h5', monitor= 'val_loss', mode= 'min', save_best_only=True, verbose=1, ),
           ModelCheckpoint(filepath= convAspectModelMetadata.newFilename() + '-high-fscore.h5', monitor= 'val_crf_fscore', mode= 'max', save_best_only=True, verbose=1, ),
           LambdaCallback(on_epoch_end = lambda epoch, logs: dump(AspectModelAuxData().ViterbiTransParams,
                                                                  convAspectModelMetadata.newFilename() + '-trans_params.joblib')),
           #EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25),
           #EarlyStopping(monitor='val_crf_fscore', mode='max', verbose=1,patience=25),
           CSVLogger(convAspectModelMetadata.newFilename() + '.log'),
           TensorBoard(log_dir = "models/tensorflow-results", histogram_freq = 0,
                       write_graph = True, write_images = True,
                       update_freq = 'epoch', profile_batch = 2, embeddings_freq = 0),
           LambdaCallback(on_epoch_end = lambda epoch, logs: convAspectModelMetadata.saveMetadata(model))]
#%% Training model

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,workers=8,
                    use_multiprocessing=True,validation_data=(x_val, y_val),callbacks=callbacks)


#%% Saving model
convAspectModelMetadata.updateMetadata(history.history)
model.save(convAspectModelMetadata.newFilename() + '.h5')
dump(AspectModelAuxData().ViterbiTransParams, convAspectModelMetadata.newFilename() + '-trans_params.joblib')
convAspectModelMetadata.saveMetadata(model)

print("Trainig complete")


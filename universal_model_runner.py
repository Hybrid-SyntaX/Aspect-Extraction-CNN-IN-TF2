import tensorflow as tf
from joblib import dump
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback, TensorBoard

from AspectModelHelperClasses import AspectModelAuxData, AspectModelMetadata, crf_fscore, crf_precision, \
    crf_recall
from DatasetReader import DatasetReader
from POSAspectModel import POSAspectModel
from helper_util import use_cpu


def read_data():
    max_sentence_length = 65
    # nostopwords="-nostopwords"

    # Load data
    embeddings_filename = r'data/sentic2vec-utf8.csv'
    restaurantDataset = DatasetReader(f'data/Restaurants_Train_v2.xml.iob',
                                      f'data/Restaurants_Test_Data_phaseB.xml.iob',
                                      'data/aspect-tags.txt',
                                      embeddings_filename,
                                      max_sentence_length)
    # x_train,y_train,x_val,y_val=restaurantDataset.prepareData()

    # Preprocess data
    return restaurantDataset


def preprocess_data(restaurantDataset):
    #x_train, x_train_pos, y_train, x_val, x_val_pos, y_val = restaurantDataset.prepareDataForPos()
    return restaurantDataset.prepareDataForPos()


def create_model(restaurantDataset):
    posAspectModel = POSAspectModel(
        num_tokens = len(restaurantDataset.word_index) + 100,
        max_sentence_length = restaurantDataset.max_sentence_length,
        num_tags = len(restaurantDataset.labels_dict))

    model = posAspectModel.createKerasModel()

    posAspectModelMetadata = AspectModelMetadata(posAspectModel)

    metric_fns = [crf_fscore, crf_precision, crf_recall]  # crf_accuracy,

    # metric_fns=[]

    optimizer = 'adam'
    epochs = 47  # original: 200 , but ran for 47 epochs
    batch_size = 30  # 30

    model.compile(optimizer = optimizer, metrics = metric_fns)

    modelMetadata = posAspectModelMetadata.createModelMetadata(metric_fns, epochs, batch_size, optimizer)

    return  model,modelMetadata


def train(model,modelMetadata ,preprocessed_data,batch_size,epochs):
    x_train, x_train_pos, y_train, x_val, x_val_pos, y_val =preprocessed_data

    callbacks = [  # LambdaCallback(on_batch_end  = lambda batch, logs: updateSentenceLengths(batch) ),
        LambdaCallback(on_epoch_end = lambda epoch, logs: modelMetadata.updateMetadata(logs)),
        ModelCheckpoint(filepath = modelMetadata.newFilename() + '-low-loss.h5', monitor = 'val_loss',
                        mode = 'min', save_best_only = True, verbose = 1, ),
        ModelCheckpoint(filepath = modelMetadata.newFilename() + '-high-fscore.h5', monitor = 'val_crf_fscore',
                        mode = 'max', save_best_only = True, verbose = 1, ),
        LambdaCallback(on_epoch_end = lambda epoch, logs: dump(AspectModelAuxData().ViterbiTransParams,
                                                               modelMetadata.newFilename() + '-trans_params.joblib')),
        # EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25),
        # EarlyStopping(monitor='val_crf_fscore', mode='max', verbose=1,patience=25),
        CSVLogger(modelMetadata.newFilename() + '.log'),
        TensorBoard(log_dir = "models/tensorflow-results", histogram_freq = 0,
                    write_graph = True, write_images = True,
                    update_freq = 'epoch', profile_batch = 2, embeddings_freq = 0),
        LambdaCallback(on_epoch_end = lambda epoch, logs: modelMetadata.saveMetadata(model))]
    # %% Training model

    history = model.fit(x_train_pos, y_train, batch_size = batch_size, epochs = epochs, workers = 8,
                        use_multiprocessing = True, validation_data = (x_val_pos, y_val), callbacks = callbacks)


def create_modelmetadata(aspectmodel):
    posAspectModelMetadata = AspectModelMetadata(aspectmodel)
    modelMetadata = posAspectModelMetadata.createModelMetadata(metric_fns, epochs, batch_size, optimizer)

    return modelMetadata

def main():
    restaurantDataset=read_data()
    preprocessed_data=preprocess_data(restaurantDataset)
    model = create_model(restaurantDataset)
    create_modelmetadata(model)
    train(model,modelMetadata ,preprocessed_data,batch_size,epochs)
    use_model()


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
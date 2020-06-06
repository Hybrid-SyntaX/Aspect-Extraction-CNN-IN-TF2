# %% Importing libraries
import argparse

import joblib
import numpy as np
import tensorflow_addons as tfa
from joblib import dump
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, LambdaCallback, TensorBoard
import tensorflow as tf

import dataset_util
from AspectModelHelperClasses import AspectModelAuxData, AspectModelMetadata, crf_accuracy, crf_fscore, crf_precision, \
    crf_recall, crf_loglikelihood_loss, KAspectModel

from ConvAspectModel import ConvAspectModel
from DatasetReader import DatasetReader
from POSAspectModel import POSAspectModel
from helper_util import use_cpu

def configure_enviroment():
    # %% Use cpu
    tf.config.set_soft_device_placement(False)
    #tf.config.experimental_run_functions_eagerly(True) #reqruied for GPU
    use_cpu(True)

#%% Reading data
def readData(max_sentence_length):
    #embeddings_filename=r'P:\Ongoing\University\Data Mining\Aspect Extraction\_\aspect-extraction-master\glove\glove.840B.300d.txt'
    embeddings_filename=r'data/sentic2vec-utf8.csv'
    restaurantDataset= DatasetReader('data/Restaurants_Train_v2.xml.iob',
                                     'data/Restaurants_Test_Data_phaseB.xml.iob',
                                     'data/aspect-tags.txt',
                                     embeddings_filename,
                                     max_sentence_length)
    return restaurantDataset

def createModel(restaurantDataset,aspectModel):
    # %% Creating model


    keras_model =aspectModel.createKerasModel()

    metric_fns=[crf_fscore,crf_precision,crf_recall]#crf_accuracy,
    metric_fns=[]
    optimizer='adam'


    keras_model.compile(optimizer=optimizer, metrics=metric_fns)

    convAspectModelMetadata= AspectModelMetadata(aspectModel,metric_fns)
    convAspectModelMetadata.modelMetadata['train_info']['optimizer'] = optimizer

    return keras_model,convAspectModelMetadata

def updateSentenceLengths(batch):
    print(batch)
    input()
    AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
    print(AspectModelAuxData().SentencesLength [0])
    input()
# %% Create callbacks

def train(model, modelMetadata, x_train, y_train, x_val, y_val, epochs, batch_size):


    modelMetadata.modelMetadata['train_info']['epochs']=epochs
    modelMetadata.modelMetadata['train_info']['batch_size'] = epochs
    model.summary()
    print(modelMetadata.modelMetadata)

    input('press to beign training!')
    callbacks=[#LambdaCallback(on_batch_end  = lambda batch, logs: updateSentenceLengths(batch) ),
                LambdaCallback(on_epoch_end = lambda epoch, logs: modelMetadata.updateMetadata(logs)),
               ModelCheckpoint(filepath= modelMetadata.newFilename() + '-low-loss.h5', monitor= 'val_loss', mode= 'min', save_best_only=True, verbose=1, ),
               ModelCheckpoint(filepath= modelMetadata.newFilename() + '-high-fscore.h5', monitor= 'val_crf_fscore', mode= 'max', save_best_only=True, verbose=1, ),
               #LambdaCallback(on_epoch_end = lambda epoch, logs: dump(AspectModelAuxData().ViterbiTransParams,convAspectModelMetadata.newFilename() + '-trans_params.joblib')),
               #EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25),
               #EarlyStopping(monitor='val_crf_fscore', mode='max', verbose=1,patience=25),
               CSVLogger(modelMetadata.newFilename() + '.log'),
               TensorBoard(log_dir = "models/tensorflow-results", histogram_freq = 0,
                           write_graph = True, write_images = True,
                           update_freq = 'epoch', profile_batch = 2, embeddings_freq = 0),
               #LambdaCallback(on_epoch_end = lambda epoch, logs: convAspectModelMetadata.saveMetadata(model))
            ]
    #%% Training model

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,workers=8,
                        use_multiprocessing=True,validation_data=(x_val, y_val),callbacks=[])

    return history

def save(model,convAspectModelMetadata,history):
    convAspectModelMetadata.updateMetadata(history.history)
    model.save(convAspectModelMetadata.newFilename() + '.h5')
    dump(AspectModelAuxData().ViterbiTransParams, convAspectModelMetadata.newFilename() + '-trans_params.joblib')
    convAspectModelMetadata.saveMetadata(model)

    print("Trainig complete")

#%% Saving model



def train_model(model_type):
    configure_enviroment()
    aspectModel=None
    history=None

    x_train, x_train_pos, y_train, x_val, x_val_pos, y_val=[],[],[],[],[],[]
    restaurantDataset= readData(max_sentence_length=65)

    if model_type == 'conv2d':
        x_train,y_train,x_val,y_val= restaurantDataset.prepareData() #prepare_data(restaurantDataset)
    elif model_type == 'pos':
        x_train, x_train_pos, y_train, x_val, x_val_pos, y_val = restaurantDataset.prepareDataForPos()

    if model_type=='conv2d':
        aspectModel = ConvAspectModel(
            embedding_matrix = restaurantDataset.embedding_matrix,
            num_tokens = len(restaurantDataset.word_index) + 100,
            max_sentence_length = restaurantDataset.max_sentence_length,
            num_tags = len(restaurantDataset.labels_dict))
    elif model_type=='pos':
        aspectModel = POSAspectModel(
            embedding_matrix = restaurantDataset.embedding_matrix,
            num_tokens = len(restaurantDataset.word_index) + 100,
            max_sentence_length = restaurantDataset.max_sentence_length,
            num_tags = len(restaurantDataset.labels_dict))


    model,convAspectModelMetadata=createModel(restaurantDataset,aspectModel = aspectModel)
    if model_type=='conv2d':
        history = train(model,convAspectModelMetadata,x_train,y_train,x_val,y_val,epochs = 200,batch_size = 30)
    elif model_type == 'pos':
        history = train(model, convAspectModelMetadata, x_train_pos, y_train, x_val_pos, y_val, epochs = 200, batch_size = 30)
    save(model,convAspectModelMetadata,history)


def evaluate_model(model_type,model_filename):
    # read data
    x_train, x_train_pos, y_train, x_val, x_val_pos, y_val = [], [], [], [], [], []
    restaurantDataset = readData(max_sentence_length = 65)

    if model_type == 'conv2d':
        x_train, y_train, x_val, y_val = restaurantDataset.prepareData()  # prepare_data(restaurantDataset)
    elif model_type == 'pos':
        x_train, x_train_pos, y_train, x_val, x_val_pos, y_val = restaurantDataset.prepareDataForPos()

    #load model
    model, trans_params = loadModel(model_filename)

    #evaluate
    results = model.evaluate_with_crf_metrics(x_train, y_train, trans_params, 'all')
    print("Train data (custom):", results)
    results = model.evaluate_with_crf_metrics(x_val, y_val, trans_params, 'all')
    print("Validation data (custom):", results)

    results = model.evaluate_with_sklearn_crf_metrics(x_train, y_train, trans_params, 'all')
    print("Train data (sklearn):", results)
    results = model.evaluate_with_sklearn_crf_metrics(x_val, y_val, trans_params, 'all')
    print("Validation data (sklearn):", results)


def loadModel(model_filename):
    model = tf.keras.models.load_model(model_filename, custom_objects = {'KAspectModel': KAspectModel}, compile = False)
    trans_params = joblib.load(model_filename.split('.')[0] + '-trans_params.joblib')
    print('trans_params loaded', trans_params)
    model.summary()
    model.compile()
    return model, trans_params


def use_model(model_type, model_filename, input_sentence):
    max_sentence_length = 65
    restaurantDataset= readData(max_sentence_length=max_sentence_length)
    x_train, x_train_pos, y_train, x_val, x_val_pos, y_val = restaurantDataset.prepareDataForPos()

    model, trans_params = loadModel(model_filename)

    test_samples = [
        input_sentence,
    ]
    input_sent,input_sentence_pos =dataset_util.prepareSentence(input_sentence,restaurantDataset.pos_tags,oneHotEncode = True)

    test_samples_pos = [
        input_sentence_pos
    ]

    test_samples_X = restaurantDataset.vectorizer(np.array([[s] for s in test_samples])).numpy()
    sentences_lengths = [np.count_nonzero(sentence) for sentence in test_samples_X]
    test_samples_X = pad_sequences(test_samples_X, maxlen = max_sentence_length, padding = 'post')

    test_samples_pos =pad_sequences(test_samples_pos,maxlen = max_sentence_length,padding = 'post')

    # print(test_samples_X)
    class_idx = {y: x for x, y in restaurantDataset.labels_dict.items()}

    logits=[]
    if model_type=='conv2d':
        logits = model.predict(test_samples_X)
    elif model_type=='pos':

        logits = model.predict(test_samples_pos)

    sequence_logits = logits[0][:sentences_lengths[0]]
    # print(sequence_logits)
    # print('trans_params:',trans_params)
    viterbi_sequence, viterbi_score = tfa.text.viterbi_decode(sequence_logits, trans_params)

    #print(len(logits))
    pred_seqs = []
    for logit_i in range(len(logits)):
    #for logit_i,logit in enumerate(logits):
        sent_len = sentences_lengths[logit_i]
        logit = logits[logit_i][:sent_len]
        viterbi_sequence, viterbi_score = tfa.text.viterbi_decode(logit, trans_params)
        pred_seq = []
        for tag in viterbi_sequence:
            pred_seq.append(class_idx[tag])
        pred_seqs.append(pred_seq)

    print(test_samples)
    print(pred_seqs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("--model", "-m", help = "Select model type")
    parser.add_argument("--train", "-t", help = "Train the model")
    parser.add_argument("--eval", "-e", help = "Evaluate the model", nargs=2)
    parser.add_argument("--use", "-u", help = "Use the model",nargs=3)
    # Read arguments from the command line
    args = parser.parse_args()

    # Check for --version or -V
    if args.version:
        print("This is myprogram version 0.1")

    if args.train:
        train_model(model_type=args.train)
    if args.eval:
        evaluate_model(model_type=args.eval[0] , model_filename=args.eval[1])

    if args.use:
        use_model(model_type = args.use[0], model_filename = args.use[1],input_sentence=args.use[2])

if __name__ == "__main__":
    main()

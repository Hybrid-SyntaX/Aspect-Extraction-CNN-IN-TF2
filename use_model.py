# %% Importing libraries
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
import helper_util
import tensorflow.keras as K
from AspectModelHelperClasses import KAspectModel
from DatasetReader import DatasetReader
print('All libraries are imported')

# %% configs
tf.config.experimental_run_functions_eagerly(True)
helper_util.use_cpu(True)

# %% read commandline arguments
model_filename= sys.argv[1]
input_sentence= sys.argv[2]

# %% load data

max_sentence_length=65
restaurantDataset= DatasetReader(r'data/Restaurants_Train_v2.xml.iob',r'data/Restaurants_Test_Data_phaseB.xml.iob','data/aspect-tags.txt','data/sentic2vec-utf8.csv',max_sentence_length)
x_train,y_train,x_val,y_val=restaurantDataset.prepareData()
#3 class model
#restaurantDataset.labels_dict.pop('NN') # remove NN class

class_names=[k for k in restaurantDataset.labels_dict.keys()]
print('intitalization complete')


# %% Loading model

model = tf.keras.models.load_model(model_filename, custom_objects={'KAspectModel':KAspectModel,'squeeze':K.backend.squeeze,'Lambda':K.layers.Lambda},compile = False)
trans_params = load(model_filename.split('.')[0]+'-trans_params.joblib')
print('trans_params loaded',trans_params)
model.summary()

# model.compile()
# results = model.evaluate_with_sklearn_crf_metrics(x_train,y_train,trans_params,'recall')
# print("Train data recall :", results)
# results = model.evaluate_with_sklearn_crf_metrics(x_val,y_val,trans_params,'recall')
# print("Validation data recall :", results)
#
# results = model.evaluate_with_sklearn_crf_metrics(x_train,y_train,trans_params,'precision')
# print("Train data precision:", results)
# results = model.evaluate_with_sklearn_crf_metrics(x_val,y_val,trans_params,'precision')
# print("Validation data precision :", results)
#
# results = model.evaluate_with_sklearn_crf_metrics(x_train,y_train,trans_params,'fscore')
# print("Train data F1 Score:", results)
# results = model.evaluate_with_sklearn_crf_metrics(x_val,y_val,trans_params,'fscore')
# print("Validation data F1 Score:", results)
# %% Testing model

test_samples = [
    input_sentence,
]

test_samples_X = restaurantDataset.vectorizer(np.array([[s] for s in test_samples])).numpy()
sentences_lengths = [np.count_nonzero(sentence) for sentence in test_samples_X]
test_samples_X= pad_sequences(test_samples_X,maxlen=max_sentence_length,padding='post')

#print(test_samples_X)
class_idx = {y:x for x,y in  restaurantDataset.labels_dict.items()}

logits = model.predict(test_samples_X)
sequence_logits = logits[0][:sentences_lengths[0]]
#print(sequence_logits)
#print('trans_params:',trans_params)
viterbi_sequence , viterbi_score = tfa.text.viterbi_decode(sequence_logits,trans_params)

pred_seqs=[]
for logit_i in range(len(logits)):
    sent_len= sentences_lengths[logit_i]
    logit=logits[logit_i][:sent_len]
    viterbi_sequence , viterbi_score = tfa.text.viterbi_decode(logit,trans_params)
    pred_seq=[]
    for tag in viterbi_sequence:
        pred_seq.append(class_idx[tag])
    pred_seqs.append(pred_seq)

print(test_samples)
print(pred_seqs) 

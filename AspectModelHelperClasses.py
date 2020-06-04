from datetime import datetime
import json
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import joblib
from sklearn.metrics import precision_recall_fscore_support


class SingletonMeta(type):
    _instance= None

    def __call__(self):
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance



class AspectModelAuxData(metaclass = SingletonMeta):
    global_trans_params = []
    sent_lens = []
    currentBatch = []

    def getGlobalTransParams(self):  return self.global_trans_params

    def setGlobalTransParams(self, val): self.global_trans_params = val

    def delGlobalTransParams(self): del self.global_trans_params

    ViterbiTransParams = property(getGlobalTransParams, setGlobalTransParams, delGlobalTransParams)

    def setSentencesLengths(self, val):  self.sent_lens = val

    def getSentencesLength(self):   return self.sent_lens

    def delSentencesLength(self):   del self.sent_lens

    SentencesLength = property(getSentencesLength, setSentencesLengths, delSentencesLength)


class AspectModelBase:
    def __init__(self):
        pass


class AspectModelMetadata:
    def __init__(self, aspectModel: AspectModelBase):

        self.aspectModel = aspectModel

    def toDict(self):
        return {
            'model_class': type(self.aspectModel).__name__,
            'learning_rate': self.aspectModel.lr,
            'dropout_rate': self.aspectModel.DROPOUT_RATE,
            'conv_droout_rate': self.aspectModel.DROPOUT_CONV,
            'filter_sizes': self.aspectModel.FILTER_SIZE,
            'feature_maps_count': self.aspectModel.NUMBER_OF_FEATURE_MAPS,
            'classes_count': self.aspectModel.NUM_TAGS,
            'window_length': self.aspectModel.WINDOW_LEN,
            'dim': self.aspectModel.DIM,
            'embedding_dim': self.aspectModel.embedding_dim,
            'word_counts': self.aspectModel.num_tokens,
            'max_sentence_length': self.aspectModel.max_sentence_length
        }

    def createModelMetadata(self, metric_fns, epochs, batch_size, optimizer):
        metrics_dict = {}
        for fn in metric_fns:
            metrics_dict[fn.__name__] = 0
            metrics_dict[f'val_{fn.__name__}'] = 0

        self.modelMetadata = {
            'model_configs': self.toDict(),
            'metrics': metrics_dict,
            'train_info': {
                'epochs': epochs,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'epochs_trained': 0,
                'last_trained': datetime.timestamp(datetime.now()),
            },
        }

        return self.modelMetadata

    def newFilename(self,modelMetadata=None):
        if modelMetadata is not None:
            self.modelMetadata=modelMetadata

        modelClass = self.modelMetadata['model_configs']['model_class']
        lastTrained = round(self.modelMetadata['train_info']['last_trained'])
        epochsTrained = self.modelMetadata['train_info']['epochs_trained']
        batchSize = self.modelMetadata['train_info']['batch_size']

        if 'val_crf_fscore' in self.modelMetadata['metrics']:
            score = round(self.modelMetadata['metrics']['val_crf_fscore']*100)
        elif 'val_loss' in self.modelMetadata['metrics']:
            score=round(self.modelMetadata['metrics']['val_loss']*100)
        else:
            score=0

        return f"models\\{modelClass}-{lastTrained}-{epochsTrained}x{batchSize}-{score}"

    def updateMetadata(self, logs):
        print('Metadata is updated')
        #for key in self.modelMetadata['metrics'].keys():
        for key in logs.keys():
            val = logs[key]
            if isinstance(val,list) and len(val) >0 :
                val=val[0]
            self.modelMetadata['metrics'][key] = val
            # modelMetadata['metrics'][f'val_{key}']=logs[f'val_{key}']
        if 'val_loss' in logs:
            val = logs['val_loss']
            if isinstance(val,list) and len(val) >0 :
                val =val [0]
            self.modelMetadata['metrics']['val_loss'] = val

        self.modelMetadata['train_info']['epochs_trained'] += 1
        self.modelMetadata['train_info']['last_trained'] = datetime.timestamp(datetime.now())

        return self.modelMetadata

    def saveMetadata(self, model):
        # Save metadata
        with open(self.newFilename() + "-metadata.json", 'w') as fp:
            json.dump(self.modelMetadata, fp)

        # Save summary
        with open(self.newFilename() + "-model-summary.txt", 'w') as fh:
            model.summary(print_fn = lambda x: fh.write(x + '\n'))

        # Save viterbi trans_params
        # joblib.dump(AspectModelAuxData().getGlobalTransParams(),
        #             self.newFilename() + '-trans_params.joblib')

        # Save model
        #model.save(self.newFilename() + '.h5')

def crf_loglikelihood_loss(y_true, y_pred):
    log_likelihood, trans_params = tfa.text.crf_log_likelihood(y_pred, y_true, AspectModelAuxData().SentencesLength)
    AspectModelAuxData().ViterbiTransParams = trans_params

    loss = tf.math.reduce_mean(-log_likelihood)  #

    return loss

class KAspectModel(K.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.global_trans_params=None

    def evaluate_with_f1(self,
               x=None,
               y=None,
               trans_params=None):
        #return super().evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,max_queue_size,workers,use_multiprocessing,return_dict)
        AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
        AspectModelAuxData().ViterbiTransParams=trans_params
        y_pred = self.predict(x)
        #print(y)
        return crf_precision_recall_fscore_manual('fscore',y,y_pred)
    # def evaluate(self,
    #            x=None,
    #            y=None,
    #            batch_size=None,
    #            verbose=1,
    #            sample_weight=None,
    #            steps=None,
    #            callbacks=None,
    #            max_queue_size=10,
    #            workers=1,
    #            use_multiprocessing=False,
    #            return_dict=False):
    #     #return super().evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,max_queue_size,workers,use_multiprocessing,return_dict)
    #     AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
    #     y_pred = self.predict(x)
    #     #print(y)
    #     return crf_precision_recall_fscore_manual('fscore',y,y_pred)

    @tf.function(experimental_compile = True)
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        # sent_len= [np.count_nonzero(sent) for sent in x]
        AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]


        with tf.GradientTape() as tape:
            y_pred = self(x, training = True)  # Forward pass


            log_likelihood, trans_params = tfa.text.crf_log_likelihood(y_pred, y, AspectModelAuxData().SentencesLength)
            AspectModelAuxData().ViterbiTransParams = trans_params

            loss = tf.math.reduce_mean(-log_likelihood)  #


            #manuily
            # losses = self.compiled_loss(y, y_pred, regularization_losses = self.losses)
            # #print(labels_pred)
            #
            # #labels_pred = tf.cast(tf.argmax(y_pred, axis = -1),tf.int32)
            # #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = labels_pred, labels = y)
            # mask = tf.sequence_mask(AspectModelAuxData().SentencesLength)
            # losses = tf.boolean_mask(losses, mask)
            # loss = tf.reduce_mean(losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def getActivationFunctionName(activation_fn):
    if hasattr(activation_fn, '__name__'):
        return activation_fn.__name__
    else:
        return activation_fn


def crf_precision_recall_fscore_support(metric_type, y_true, y_pred):
    gold = []
    pred = []

    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        gold.extend(lab)
        pred.extend(viterbi_seq)

    gold = K.utils.to_categorical(gold, 4)
    pred = K.utils.to_categorical(pred, 4)

    if len(gold) > 0 and len(pred) > 0:
        precision, recall, fscore, support = precision_recall_fscore_support(gold, pred, average = 'macro',
                                                                             zero_division = 0)  # average='macrp'
    else:
        precision, recall, fscore, support = 0, 0, 0, 0

    if metric_type == 'precision':
        return precision
    elif metric_type == 'recall':
        return recall
    elif metric_type == 'fscore':
        return fscore
    elif metric_type == 'support':
        return support


def decodeViterbi(y_true, y_pred):
    labs = []
    lab_preds = []
    viterbi_seqs = []
    for lab, lab_pred, length in zip(y_true, y_pred, AspectModelAuxData().SentencesLength):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        viterbi_seq, viterbi_score = tfa.text.viterbi_decode(lab_pred, AspectModelAuxData().ViterbiTransParams)

        yield lab, lab_pred, viterbi_seq


@tf.function(experimental_compile=True)
def crf_accuracy(y_true, y_pred):
    accs = []
    #accs=tf.TensorArray()
    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        #result = tf.math.equal(lab, viterbi_seq)
        #accs=tf.concat(accs,result)
        #accs +=result
        accs += [a == b for (a, b) in zip(lab, viterbi_seq)]
    return np.mean(accs)

from seqeval.metrics import f1_score

@tf.function(experimental_compile=True)
def crf_precision_recall_fscore_manual(metric_type, y_true, y_pred):
    #gold = []
    #pred = []
    correct_preds, total_correct, total_preds = 0., 0., 0.

    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        #result = lab & viterbi_seq
        #result = tf.math.subtract(lab,viterbi_seq)
        result = tf.math.equal(lab,viterbi_seq)

        correct_preds +=  np.count_nonzero(result) #tf.math.count_nonzero(result) #tf.reduce_sum(tf.cast(np.count_nonzero(result), tf.float32))
        total_preds += len(viterbi_seq)
        total_correct += len(lab)

        #result = tf.math.logical_and(lab,viterbi_seq)
        # print('Result:', result )
        # print('correct_preds:', correct_preds)
        # print('total_preds:', total_preds)
        # print('total_correct:', total_correct)
        # input()

    #     gold.extend(lab)
    #     pred.extend(viterbi_seq)
    #
    # gold = K.utils.to_categorical(gold, 4)
    # pred = K.utils.to_categorical(pred, 4)

    precision = correct_preds / total_preds if correct_preds > 0 else 0
    recall = correct_preds / total_correct if correct_preds > 0 else 0
    fscore = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0


    #precision, recall, fscore, support = precision_recall_fscore_support(gold, pred, average = 'macro',
    #                                                                          zero_division = 0)
    #
    # if len(gold) > 0 and len(pred) > 0:
    #     precision, recall, fscore, support = precision_recall_fscore_support(gold, pred, average = 'macro',
    #                                                                          zero_division = 0)  # average='macrp'
    # else:
    #     precision, recall, fscore, support = 0, 0, 0, 0

    # print('f1:',f1)
    # input()
    if metric_type == 'precision':
        return precision
    elif metric_type == 'recall':
        return recall
    elif metric_type == 'fscore':
        return fscore


#def crf_fscore(y_true, y_pred): return crf_precision_recall_fscore_support('fscore', y_true, y_pred)
#def crf_precision(y_true, y_pred): return crf_precision_recall_fscore_support('precision', y_true, y_pred)
#def crf_recall(y_true, y_pred): return crf_precision_recall_fscore_support('recall', y_true, y_pred)
def crf_support(y_true, y_pred): return crf_precision_recall_fscore_support('support', y_true, y_pred)

def crf_fscore(y_true, y_pred): return crf_precision_recall_fscore_manual('fscore',y_true, y_pred)
def crf_precision(y_true, y_pred): return crf_precision_recall_fscore_manual('precision',y_true, y_pred)
def crf_recall(y_true, y_pred): return crf_precision_recall_fscore_manual('fscore',y_true, y_pred)
#def crf_fscore(y_true, y_pred): return crf_precision_recall_fscore_manual('fscore',y_true, y_pred)

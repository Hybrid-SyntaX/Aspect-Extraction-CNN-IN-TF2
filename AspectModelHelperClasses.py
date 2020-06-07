from datetime import datetime
import json


import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import joblib
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import f1_score, accuracy_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
import seqeval

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
    def toDict(self):
        pass
    def __init__(self):
        pass


class AspectModelMetadata:
    def __init__(self, aspectModel: AspectModelBase,metric_fns):

        self.aspectModel = aspectModel

        metrics_dict = {}
        for fn in metric_fns:
            metrics_dict[fn.__name__] = 0
            metrics_dict[f'val_{fn.__name__}'] = 0

        self.modelMetadata = {
            'model_configs': self.aspectModel.toDict(),
            'metrics': metrics_dict,
            'train_info': {
                'epochs': 0,
                'batch_size': 0,
                'optimizer': None,
                'epochs_trained': 0,
                'last_trained': datetime.timestamp(datetime.now()),
            },
        }

    def createModelMetadata(self, metric_fns, epochs, batch_size, optimizer):
        metrics_dict = {}
        for fn in metric_fns:
            metrics_dict[fn.__name__] = 0
            metrics_dict[f'val_{fn.__name__}'] = 0

        self.modelMetadata = {
            'model_configs': self.aspectModel.toDict(),
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
        #print('Metadata is updated')
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
class CRFMetricsData(metaclass = SingletonMeta):
    correct_preds, total_correct, total_preds = 0., 0., 0.
    isUpdated=False
    precision,recall,fscore=0.0,0.0,0.0
    i=0
    r=0
    def reset(self):
        #print(f"RESET IS CALLED {self.r} times", )
        #self.i+=1
        self.r+=1
        self.precision=0.0
        self.recall=0.0
        self.fscore=0.0


def evaluate_with_sklearn_crf_metrics(model,
                                      x=None,
                                      y=None,
                                      trans_params=None,
                                      metric_fn='fscore',useReport=False):
    #return super().evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,max_queue_size,workers,use_multiprocessing,return_dict)
    AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
    AspectModelAuxData().ViterbiTransParams=trans_params
    y_pred = model.predict(x)
    #print(y)

    if useReport:
        return crf_sklearn_classification_report(y,y_pred)
    else:
        return crf_precision_recall_fscore_support(metric_fn,y,y_pred)
def evaluate_with_crf_metrics(model,
           x=None,
           y=None,
           trans_params=None,
           metric_fn='fscore'):
    #return super().evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,max_queue_size,workers,use_multiprocessing,return_dict)
    AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
    AspectModelAuxData().ViterbiTransParams=trans_params
    y_pred = model.predict(x)
    #print(y)
    return crf_precision_recall_fscore_manual(metric_fn,y,y_pred)
def evaluate_with_crf_metrics_by_sentence(model,
           x=None,
           y=None,
           trans_params=None,
           metric_fn='fscore'):
    #return super().evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,max_queue_size,workers,use_multiprocessing,return_dict)
    AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
    AspectModelAuxData().ViterbiTransParams=trans_params
    y_pred = model.predict(x)
    #print(y)
    return crf_precision_recall_fscore_manual_by_sentence(metric_fn,y,y_pred)
def evaluate_with_crf_metrics_seqeval(model,
           x=None,
           y=None,
           trans_params=None,
           metric_fn='fscore',useReport=False):
    #return super().evaluate(x,y,batch_size,verbose,sample_weight,steps,callbacks,max_queue_size,workers,use_multiprocessing,return_dict)
    AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
    AspectModelAuxData().ViterbiTransParams=trans_params
    y_pred = model.predict(x)
    #print(y)
    if useReport:
        return seqeval_classification_report(y,y_pred)
    else:
        return crf_precision_recall_fscore_seqeval(metric_fn,y,y_pred)

class KAspectModel(K.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.global_trans_params=None
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


    def _updateSentLen(self,x):
        #AspectModelAuxData().SentencesLength.append(x)
        #tf.concat(AspectModelAuxData().SentencesLength,tf.math.count_nonzero(x))
        pass

    @tf.function(experimental_compile = True, experimental_relax_shapes = True)
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        AspectModelAuxData().SentencesLength = [np.count_nonzero(sent) for sent in x]
        #self._updateSentLen(x)

        with tf.GradientTape() as tape:
            #CRFMetricsData().reset()
            y_pred = self(x, training = True)  # Forward pass

            if AspectModelAuxData().ViterbiTransParams !=[]:
                log_likelihood, trans_params = tfa.text.crf_log_likelihood(y_pred, y, AspectModelAuxData().SentencesLength,AspectModelAuxData().ViterbiTransParams)
            else:
                log_likelihood, trans_params = tfa.text.crf_log_likelihood(y_pred, y, AspectModelAuxData().SentencesLength)
            AspectModelAuxData().ViterbiTransParams = trans_params

            loss = tf.math.reduce_mean(-log_likelihood)  #


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


#@tf.function(experimental_compile=True)
def crf_sklearn_classification_report(y_true, y_pred):
    gold = []
    pred = []
    accs=[]
    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        accs += [a == b for (a, b) in zip(lab, viterbi_seq)]
        gold.extend(lab)
        pred.extend(viterbi_seq)

    gold = K.utils.to_categorical(gold, 3)
    pred = K.utils.to_categorical(pred, 3)

    from sklearn.metrics import classification_report
    return classification_report(gold,pred,digits = 4)

def crf_precision_recall_fscore_support(metric_type, y_true, y_pred):
    gold = []
    pred = []
    accs=[]
    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        accs += [a == b for (a, b) in zip(lab, viterbi_seq)]
        gold.extend(lab)
        pred.extend(viterbi_seq)

    gold = K.utils.to_categorical(gold, 3)
    pred = K.utils.to_categorical(pred, 3)

    if len(gold) > 0 and len(pred) > 0:
        precision, recall, fscore, support = precision_recall_fscore_support(gold, pred, average = 'macro',
                                                                             zero_division = 0)  # average='macrp'
    else:
        precision, recall, fscore, support = 0, 0, 0, 0

    accuracy=np.mean(accs)
    if metric_type == 'precision':
        return precision
    elif metric_type == 'recall':
        return recall
    elif metric_type == 'fscore':
        return fscore
    elif metric_type == 'support':
        return support
    elif metric_type == 'accuracy':
        return accuracy
    elif metric_type=='all':
        return {'precision':precision,'recall':recall,'fscore':fscore,'support':support,'accuracy':accuracy}
tags_idx=['I-A','O','B-A']#{0:'I-A',1:'O',2:'B-A'}


def seqeval_classification_report(y_true, y_pred):
    gold = []
    pred = []
    accs = []
    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        # accs += [a == b for (a, b) in zip(lab, viterbi_seq)]
        lab = [tags_idx[i] for i in lab]
        viterbi_seq = [tags_idx[i] for i in viterbi_seq]

        gold.extend(lab)
        pred.extend(viterbi_seq)

    return seqeval.metrics.classification_report(gold,pred,digits = 4)
def crf_precision_recall_fscore_seqeval(metric_type, y_true, y_pred):
    gold = []
    pred = []
    accs=[]
    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        #accs += [a == b for (a, b) in zip(lab, viterbi_seq)]
        lab = [tags_idx[i] for i in lab]
        viterbi_seq = [tags_idx[i] for i in viterbi_seq]

        gold.extend(lab)
        pred.extend(viterbi_seq)


    # gold = K.utils.to_categorical(gold, 3)
    # pred = K.utils.to_categorical(pred, 3)

    if len(gold) > 0 and len(pred) > 0:
        # precision, recall, fscore, support = precision_recall_fscore_support(gold, pred, average = 'macro',
        #
        #
        #                                                                      azero_division = 0)  # average='macrp'
        support=0
        accuracy=accuracy_score(gold,pred)
        fscore=f1_score(gold,pred)
        precision=precision_score(gold,pred)
        recall = recall_score(gold, pred)
    else:
        precision, recall, fscore, support = 0, 0, 0, 0

    accuracy=np.mean(accs)
    if metric_type == 'precision':
        return precision
    elif metric_type == 'recall':
        return recall
    elif metric_type == 'fscore':
        return fscore
    elif metric_type == 'support':
        return support
    elif metric_type == 'accuracy':
        return accuracy
    elif metric_type=='all':
        return {'precision':precision,'recall':recall,'fscore':fscore,'support':support,'accuracy':accuracy}
#@tf.function(experimental_compile=True)
def crf_precision_recall_fscore_manual(metric_type, y_true, y_pred):

    correct_preds, total_correct, total_preds = 0., 0., 0.

    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):

        result = tf.math.equal(lab,viterbi_seq)

        correct_preds +=  np.count_nonzero(result) # TP + TN
        total_preds += len(viterbi_seq)
        total_correct += len(lab)

    accuracy = correct_preds / total_preds
    precision = correct_preds / total_preds if correct_preds > 0 else 0
    recall = correct_preds / total_correct if correct_preds > 0 else 0
    fscore = (2 * precision * recall) / (precision + recall) if correct_preds > 0 else 0

    # print('correct_preds', correct_preds)
    # print('total_preds', total_preds)
    # print('total_correct', total_correct)
    # print('f1', fscore)
    # print('precision:', precision)
    # print('recall:', recall)
    # input()

    if metric_type == 'precision':
        return precision
    elif metric_type == 'recall':
        return recall
    elif metric_type == 'fscore':
        return fscore
    elif metric_type == 'accuracy':
        return accuracy
    elif metric_type=='all':
        return {'precision':precision,'recall':recall,'fscore':fscore,'accuracy':accuracy}

    def crf_precision_recall_fscore_manual(metric_type, y_true, y_pred):

        correct_preds, total_correct, total_preds = 0., 0., 0.

        for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
            result = tf.math.equal(lab, viterbi_seq)

            correct_preds += np.count_nonzero(result)  # TP + TN
            total_preds += len(viterbi_seq)
            total_correct += len(lab)

        accuracy = correct_preds / total_preds
        precision = correct_preds / total_preds if correct_preds > 0 else 0
        recall = correct_preds / total_correct if correct_preds > 0 else 0
        fscore = (2 * precision * recall) / (precision + recall) if correct_preds > 0 else 0

        # print('correct_preds', correct_preds)
        # print('total_preds', total_preds)
        # print('total_correct', total_correct)
        # print('f1', fscore)
        # print('precision:', precision)
        # print('recall:', recall)
        # input()

        if metric_type == 'precision':
            return precision
        elif metric_type == 'recall':
            return recall
        elif metric_type == 'fscore':
            return fscore
        elif metric_type == 'accuracy':
            return accuracy
        elif metric_type == 'all':
            return {'precision': precision, 'recall': recall, 'fscore': fscore, 'accuracy': accuracy}
    #print(f'crf manual claled {CRFMetricsData().i} times')
    # CRFMetricsData().i+=1
    # CRFMetricsData().precision=precision
    # CRFMetricsData().recall = recall
    # CRFMetricsData().fscore = fscore
def crf_precision_recall_fscore_manual_by_sentence(metric_type, y_true, y_pred):

    correct_preds, total_correct, total_preds = 0., 0., 0.

    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        result = tf.math.equal(lab,viterbi_seq)
        result = tf.reduce_all(result)

        if result:
            correct_preds+=1

        total_preds += 1
        total_correct += 1

    accuracy = correct_preds/total_preds
    precision = correct_preds / total_preds if correct_preds > 0 else 0
    recall = correct_preds / total_correct if correct_preds > 0 else 0
    fscore = (2 * precision * recall) / (precision + recall) if correct_preds > 0 else 0


    if metric_type == 'precision':
        return precision
    elif metric_type == 'recall':
        return recall
    elif metric_type == 'fscore':
        return fscore
    elif metric_type == 'accuracy':
        return accuracy
    elif metric_type=='all':
        return {'precision':precision,'recall':recall,'fscore':fscore,'accuracy':accuracy}

def crf_precision_recall_fscore_manual_with_confusion_matrix(metric_type, y_true, y_pred):

    correct_preds, total_correct, total_preds = 0., 0., 0.

    tp,tn,fp,fn=0,0,0,0

    for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
        result = tf.math.equal(lab,viterbi_seq)
        result = tf.reduce_all(result)

        if result:
            tp+=1
            correct_preds+=1


        total_preds += 1
        total_correct += 1

    accuracy = correct_preds/total_preds
    precision = correct_preds / total_preds if correct_preds > 0 else 0
    recall = correct_preds / total_correct if correct_preds > 0 else 0
    fscore = (2 * precision * recall) / (precision + recall) if correct_preds > 0 else 0


    if metric_type == 'precision':
        return precision
    elif metric_type == 'recall':
        return recall
    elif metric_type == 'fscore':
        return fscore
    elif metric_type == 'accuracy':
        return accuracy
    elif metric_type=='all':
        return {'precision':precision,'recall':recall,'fscore':fscore,'accuracy':accuracy}


def crf_fscore(y_true, y_pred):
    return crf_precision_recall_fscore_manual('fscore', y_true, y_pred)

def crf_precision(y_true, y_pred):
    return crf_precision_recall_fscore_manual('precision', y_true, y_pred)

def crf_recall(y_true, y_pred):
    return crf_precision_recall_fscore_manual('recall', y_true, y_pred)


def crf_support(y_true, y_pred): return crf_precision_recall_fscore_support('support', y_true, y_pred)
#


#@tf.function(experimental_compile=True)

# def _crf_metrics_prepare(y_pred, y_true):
#
#     correct_preds, total_correct, total_preds = 0., 0., 0.
#     for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
#         result = tf.math.equal(lab, viterbi_seq)
#
#         correct_preds += np.count_nonzero(
#             result)  # tf.math.count_nonzero(result) #tf.reduce_sum(tf.cast(np.count_nonzero(result), tf.float32))
#         total_preds += len(viterbi_seq)
#         total_correct += len(lab)
#     return correct_preds, total_correct, total_preds
#
# def crf_metrics_prepare(y_pred, y_true):
#
#     #if CRFMetricsData().correct_preds==0 and  CRFMetricsData().total_correct==0 and CRFMetricsData().total_preds==0:
#     #correct_preds, total_correct, total_preds = 0., 0., 0.
#     if CRFMetricsData().isUpdated==False:
#         for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
#             result = tf.math.equal(lab, viterbi_seq)
#
#             CRFMetricsData().correct_preds += np.count_nonzero(result)  # tf.math.count_nonzero(result) #tf.reduce_sum(tf.cast(np.count_nonzero(result), tf.float32))
#             CRFMetricsData().total_preds += len(viterbi_seq)
#             CRFMetricsData().total_correct += len(lab)
#
#             CRFMetricsData().isUpdated=False
#     return CRFMetricsData().correct_preds, CRFMetricsData().total_correct, CRFMetricsData().total_preds
# @tf.function(experimental_compile=True)
# def crf_fscore(y_true, y_pred):
#     correct_preds, total_correct, total_preds = 0., 0., 0.
#
#     for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
#
#         result = tf.math.equal(lab,viterbi_seq)
#
#         correct_preds +=  np.count_nonzero(result) #tf.math.count_nonzero(result) #tf.reduce_sum(tf.cast(np.count_nonzero(result), tf.float32))
#         total_preds += len(viterbi_seq)
#         total_correct += len(lab)
#
#
#
#     precision = correct_preds / total_preds if correct_preds > 0 else 0
#     recall = correct_preds / total_correct if correct_preds > 0 else 0
#     fscore = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0
#
#     return fscore
#
#
#
# @tf.function(experimental_compile=True)
# def crf_precision(y_true, y_pred):
#     correct_preds, total_correct, total_preds = 0., 0., 0.
#
#     for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
#
#         result = tf.math.equal(lab,viterbi_seq)
#
#         correct_preds +=  np.count_nonzero(result) #tf.math.count_nonzero(result) #tf.reduce_sum(tf.cast(np.count_nonzero(result), tf.float32))
#         total_preds += len(viterbi_seq)
#         total_correct += len(lab)
#
#
#
#     precision = correct_preds / total_preds if correct_preds > 0 else 0
#     recall = correct_preds / total_correct if correct_preds > 0 else 0
#
#     return precision
#
# @tf.function(experimental_compile=True)
# def crf_recall(y_true, y_pred):
#     correct_preds, total_correct, total_preds = 0., 0., 0.
#
#     for lab, lab_pred, viterbi_seq in decodeViterbi(y_true, y_pred):
#         result = tf.math.equal(lab, viterbi_seq)
#
#         correct_preds += np.count_nonzero(
#             result)  # tf.math.count_nonzero(result) #tf.reduce_sum(tf.cast(np.count_nonzero(result), tf.float32))
#         total_preds += len(viterbi_seq)
#         total_correct += len(lab)
#
#     recall = correct_preds / total_correct if correct_preds > 0 else 0
#
#     return recall

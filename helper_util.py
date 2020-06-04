import numpy
import os
from joblib import dump, load
import tensorflow as tf
def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(numpy.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return numpy.array(cat_sequences)

def my_vectorize(train_sentences,word_index):
    train_sentences_X=[]
    for s in train_sentences:
        s_int = []
        for w in s:
                s_int.append(word_index[w.lower()])
    
        train_sentences_X.append(s_int)
def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

def loadOrCalculate(filename,func=None, *args, **kwargs):
    method =filename.split('.')[-1]
    if os.path.exists(filename):
        if method=='joblib':
            result=load(filename)
        elif method =='npz':
            data=numpy.load(filename,allow_pickle=True)
            result=data['data']
            
        print('Method output loaded')
    else:
        result =func(*args, **kwargs)
        if method=='joblib':
            dump(result,filename)
        elif method =='npz':
            numpy.savez(filename,data=result)

        print('Method calculated calculated')
    return result

def use_cpu(use):
    if use:
        tf.config.experimental_run_functions_eagerly(True)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
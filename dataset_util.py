from nltk.corpus.reader import ConllChunkCorpusReader
import numpy as np 
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
from joblib import dump, load
import os 


def readData(iob_filename):
    # Initailizing
    #noun, verb, adjective, adverb, preposition, conjunction
    pos_tags =('NOUN','VERB','ADV','ADJ','IN','CONJ')
    aspect_tags = ('I-A','O','B-A', 'NN')
    reader = ConllChunkCorpusReader('.', iob_filename,pos_tags+aspect_tags,encoding='utf-8')
    #('NP','VP','ADJP','ADVP','PNP','SBAR')

    
    #reader.chunked_words() 
    reader.chunked_sents()
    iob_sentences = reader.iob_sents() 
    #iob_words = reader.iob_words() 
    return iob_sentences


def readTags(filename):
    labels_dict={}
    class_names = []
    class_index = 0
    labels = []
    #labels_dict['-PAD-']=class_index
    with open(filename) as f:
        for c in f.readlines():
            class_names.append(c.strip())
            labels.append(class_index)
            
            #labels_dict[c.strip()] = keras.utils.to_categorical(class_index,4)
            labels_dict[c.strip()] = class_index#+1
            class_index+=1

        return labels_dict

def buildSentencesString(setnences):
    setnencesStrings=[]
    for sentence in setnences:
        setnencesStrings.append(' '.join(sentence))
    return np.array(setnencesStrings)

def buildSentenceLengths(setnences):
    sentences_lengths=[]
    for sentence in setnences:
        length=0
        for word in sentence:
            length+=len(word)
        sentences_lengths.append(length)
    return sentences_lengths

def buildSentenceLengthsByWords(setnences):
    sentences_lengths=[]
    for sentence in setnences:
        sentences_lengths.append(len(sentence.split()))
    return sentences_lengths

def createData(iob_data,labels_dict):
    sentences=[]
    pos_tags=[]
    labels=[]
    for s in iob_data:
        sentence = []
        pos_tag=[]
        sentence_label = []
        for word,pos,tag in s:
            sentence.append(word)
            pos_tag.append(pos)
            sentence_label.append(np.int32(labels_dict[tag]))

        sentences.append(sentence)
        pos_tags.append(pos_tag)
        labels.append(sentence_label)

    return np.array(sentences),pos_tags,labels

def getVectorizer(data,max_sentence_length):
    vectorizer = TextVectorization(max_tokens=50000, output_sequence_length=max_sentence_length,standardize='lower_and_strip_punctuation')
    text_ds = tf.data.Dataset.from_tensor_slices(data).batch(30)
    vectorizer.adapt(text_ds)

    return vectorizer

def getWordIndex(vectorizer):
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(2, len(voc))))

    return word_index

def readEmbeddings(filename):
    #with np.load('sentic2vec-utf8_trimmed.npz') as data:
    #        embeddings_index = data["embeddings"]
    path_to_glove_file =filename
    embeddings_index = {}
    with open(path_to_glove_file,encoding='utf-8') as f:
        for line in f.readlines():
            #word, coefs = line.split(maxsplit=1)
            row = line.strip().split(',')
            word = row[0]
            try:
                coefs = [float(x) for x in row[1:]]
                #coefs = float(np.fromstring(coefs, "f", sep=","))
            except:
                continue
            
            embeddings_index[word] = coefs

        return embeddings_index

def getEmbeddingDim(embeddings_index):
    return len(embeddings_index[next(iter(embeddings_index))])        

def createEmbeddingsMatrix(embeddings_index,word_index):
    num_tokens = len(word_index)+100
    embedding_dim =getEmbeddingDim(embeddings_index)
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word.decode("utf-8"))
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix



def get_pos_group(pos_tag,pos_tags):
    for i,(tag_group, tags) in enumerate(pos_tags.items()):
        if pos_tag in tags:
            return i,tag_group
    else:
        return -1,None

def createOneHotCodedPOSFeatures(x_train_pos,pos_tags):
    x_train_pos_features=[]
    for tagged_sentence in x_train_pos: #sentence
        tagged_sentence_onehot=[]
        for word_pos in tagged_sentence:
            i,group_name=get_pos_group(word_pos,pos_tags)
            if group_name is not None:
                tagged_sentence_onehot.append(np.eye(1, 6, i)[0])
            else:
                tagged_sentence_onehot.append(None)
        x_train_pos_features.append(tagged_sentence_onehot)

    return x_train_pos_features
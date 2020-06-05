from tensorflow.keras.preprocessing.sequence import pad_sequences
import dataset_util
import numpy as np
class DatasetReader():
    def __init__(self,train_filename,test_filename,tags_filename,embeddings_filename,max_sentence_length=None):

        self.max_sentence_length=max_sentence_length
        #train_iob = dataset_util.readData(r'Restaurants_Train_v2.xml-nostopwords.iob')
        #test_iob = dataset_util.readData(r'Restaurants_Test_Data_phaseB.xml-nostopwords.iob')
        self.train_iob = dataset_util.readData(train_filename)
        self.test_iob = dataset_util.readData(test_filename)


        #labels_dict = dataset_util.readTags('aspect-tags.txt')
        self.labels_dict = dataset_util.readTags(tags_filename)

        self.train_samples,self.train_pos_tags,self.train_labels = dataset_util.createData(self.train_iob,self.labels_dict)
        self.val_samples, self.val_pos_tags,self.val_labels = dataset_util.createData(self.test_iob,self.labels_dict)

        self.train_samples = dataset_util.buildSentencesString(self.train_samples)
        self.val_samples = dataset_util.buildSentencesString(self.val_samples)

        self.train_samples_lengths =dataset_util.buildSentenceLengthsByWords(self.train_samples)
        self.val_samples_lengths =dataset_util.buildSentenceLengthsByWords(self.val_samples)

        if self.max_sentence_length is None:
            self.max_sentence_length = int(np.max(self.train_samples_lengths)  if np.max(self.train_samples_lengths) > np.max(self.val_samples_lengths) else np.max(self.val_samples_lengths))

        self.vectorizer =  dataset_util.getVectorizer(self.train_samples,self.max_sentence_length)
        self.word_index =  dataset_util.getWordIndex(self.vectorizer)

        self.embeddings_index= dataset_util.readEmbeddings(embeddings_filename)
        self.embedding_matrix =dataset_util.createEmbeddingsMatrix(self.embeddings_index,self.word_index)

    def prepareData(self):
        x_train = self.vectorizer(np.array([[s] for s in self.train_samples])).numpy()
        x_train = pad_sequences(x_train,maxlen=self.max_sentence_length,padding='post')

        x_val = self.vectorizer(np.array([[s] for s in self.val_samples])).numpy()
        x_val =  pad_sequences(x_val,maxlen=self.max_sentence_length,padding='post')

        y_train = pad_sequences(np.asarray(self.train_labels),maxlen=self.max_sentence_length,padding='post')
        y_val = pad_sequences(np.asarray(self.val_labels),maxlen=self.max_sentence_length,padding='post')

        return x_train,y_train,x_val,y_val
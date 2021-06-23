from tensorflow.keras.preprocessing.sequence import pad_sequences
import dataset_util
import numpy as np
class DatasetReader():
    def __init__(self,train_filename,test_filename,tags_filename,embeddings_filename,max_sentence_length=None):

        self.max_sentence_length=max_sentence_length

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

        self.class_idx = {y: x for x, y in self.labels_dict.items()}

    def prepareData(self):
        x_train = self.vectorizer(np.array([[s] for s in self.train_samples])).numpy()
        x_train = pad_sequences(x_train,maxlen=self.max_sentence_length,padding='post')

        x_val = self.vectorizer(np.array([[s] for s in self.val_samples])).numpy()
        x_val =  pad_sequences(x_val,maxlen=self.max_sentence_length,padding='post')

        y_train = pad_sequences(np.asarray(self.train_labels),maxlen=self.max_sentence_length,padding='post')
        y_val = pad_sequences(np.asarray(self.val_labels),maxlen=self.max_sentence_length,padding='post')

        return x_train,y_train,x_val,y_val

    def removeUnusedFeatures(self, x, x_pos_1hot, y):
        new_sentences = []
        new_sentence_poses = []
        new_sentence_labels = []
        for sentence, sentence_pos_1hot, sentence_label in zip(x, x_pos_1hot, y):
            new_sentence = []
            new_sentence_pos = []
            new_sentence_label = []
            for word, word_pos, word_Label in zip(sentence, sentence_pos_1hot, sentence_label):
                if word_pos is not None:
                    new_sentence.append(word)
                    new_sentence_pos.append(word_pos)
                    new_sentence_label.append(word_Label)
            new_sentences.append(new_sentence)
            new_sentence_poses.append(new_sentence_pos)
            new_sentence_labels.append(new_sentence_label)

        return new_sentences, new_sentence_poses, new_sentence_labels
    pos_tags = {
        'NOUN': ('NN', 'NNS', 'NNP', 'NNPS'),
        'VERB': ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'),
        'ADV': ('RB', 'RBR', 'RBS'),
        'ADJ': ('JJ', 'JJR', 'JJS'),
        'IN': ('IN'),  # prepositions
        'CONJ': ('CC')
    }
    def prepareDataForPos(self):

        x_train, y_train, x_val, y_val = self.prepareData()

        x_train_pos = self.train_pos_tags
        x_val_pos = self.val_pos_tags

        x_train_pos_onehot = dataset_util.createOneHotCodedPOSFeatures(x_train_pos, self.pos_tags)
        x_val_pos_onehot = dataset_util.createOneHotCodedPOSFeatures(x_val_pos, self.pos_tags)

        train = self.removeUnusedFeatures(x_train,x_train_pos_onehot,y_train)
        test=self.removeUnusedFeatures(x_val,x_val_pos_onehot,y_val)
        x_train, x_train_pos, y_train = train
        x_val, x_val_pos, y_val = test

        x_train = self.vectorizer(np.array([[s] for s in self.train_samples])).numpy()
        x_train = pad_sequences(x_train, maxlen = self.max_sentence_length, padding = 'post')
        x_train_pos = pad_sequences(x_train_pos, maxlen = self.max_sentence_length, padding = 'post')

        x_val = self.vectorizer(np.array([[s] for s in self.val_samples])).numpy()
        x_val = pad_sequences(x_val, maxlen = self.max_sentence_length, padding = 'post')
        x_val_pos = pad_sequences(x_val_pos, maxlen = self.max_sentence_length, padding = 'post')

        y_train = pad_sequences(np.asarray(self.train_labels), maxlen = self.max_sentence_length, padding = 'post')
        y_val = pad_sequences(np.asarray(self.val_labels), maxlen = self.max_sentence_length, padding = 'post')

        return x_train,x_train_pos,y_train, x_val,x_val_pos,y_val
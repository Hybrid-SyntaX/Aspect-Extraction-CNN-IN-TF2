import unittest

import nltk
import numpy as np

import dataset_util
from DatasetReader import DatasetReader


class MyTestCase(unittest.TestCase):
    def prepare_data_for_pos(self):
        max_sentence_length=65
        embeddings_filename = r'data/sentic2vec-utf8.csv'
        restaurantDataset = DatasetReader(f'data/test_Restaurants_Train_Data.iob',
                                          f'data/test_Restaurants_Test_Data.iob',
                                          'data/aspect-tags.txt',
                                          embeddings_filename,
                                          max_sentence_length)
        # x_train,y_train,x_val,y_val=restaurantDataset.prepareData()

        # Preprocess data
        x_train, x_train_pos, y_train, x_val, x_val_pos, y_val = restaurantDataset.prepareDataForPos()

        self.assertEqual(np.shape(x_train),(11,max_sentence_length))
        self.assertEqual(np.shape(x_train_pos), (11, max_sentence_length,6))
        self.assertEqual(np.shape(y_train), (11, max_sentence_length))

    def test_extract_pos(self):
        input_sentence = 'I love cake'
        input_sentence_pos=dataset_util.extractPOS(input_sentence)
        self.assertEqual(input_sentence_pos, [('I', 'PRP'), ('love', 'VBP'), ('cake', 'NN')])




    def test_prepare_input(self):
        input_sentence = 'I love cake'

        sentence, sentence_pos = dataset_util.prepareSentence(input_sentence)

        self.assertEqual(len(sentence_pos),len(sentence))
        self.assertEqual(sentence,['I','love','cake'])
        self.assertEqual(sentence_pos, ['PRP', 'VBP', 'NN'])

    pos_tags = {
        'NOUN': ('NN', 'NNS', 'NNP', 'NNPS'),
        'VERB': ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'),
        'ADV': ('RB', 'RBR', 'RBS'),
        'ADJ': ('JJ', 'JJR', 'JJS'),
        'IN': ('IN'),  # prepositions
        'CONJ': ('CC')
    }
    def test_prepare_input_with_pos_groups(self):
        input_sentence = 'I love cake'

        sentence, sentence_pos = dataset_util.prepareSentence(input_sentence,self.pos_tags)

        self.assertEqual(len(sentence_pos),len(sentence))
        self.assertEqual(sentence,['I','love','cake'])
        self.assertEqual(sentence_pos, [None, 'VERB', 'NOUN'])

    def test_prepare_input_with_pos_groups_for_list(self):
        input_sentences = ['I love cake','Pizza is delicious']

        result = list(map(lambda x: dataset_util.prepareSentence(x,self.pos_tags) ,input_sentences))


        self.assertEqual(len(result),2)
        self.assertEqual((['I','love','cake'],[None, 'VERB', 'NOUN']),result[0])
        self.assertEqual((['Pizza', 'is', 'delicious'], ['NOUN', 'VERB', 'ADJ']), result[1])
        #self.assertEqual(sentence_pos, [None, 'VERB', 'NOUN'])

if __name__ == '__main__':
    unittest.main()

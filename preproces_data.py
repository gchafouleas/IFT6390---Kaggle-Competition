
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl

wordnet_lemmatizer = WordNetLemmatizer()

class PreprocessData:
    def __init__(self, train_data):
        self.vocab = list(np.unique(np.asarray(self.load_pickle_file('vocab'))))
        self.hot_one_encode_labels(train_data)

    def generate_vocabulary(self, data):
        print("Generating vocabulary")
        vocab = self.process_input(data)
        vocab = self.lemmatize_words(vocab)
        print(vocab)
        self.save_pickle_file(vocab,'vocab')
        print("Vocabulary saved")

    #reference code from : https://gist.github.com/amnrzv/596ba910524e0b1b4e8fa2167fd773bf#file-a_language_analysis-py-L2
    def process_input(self, data):
        vocab = []
        for sentence in data[0]:
            if sentence.strip():
                words = word_tokenize(sentence)
                for tag in nltk.pos_tag(words):
                    # eliminating unnecessary POS tags
                    match = re.search('\w.*', tag[1])
                    if match:
                        vocab.append(tag)
        return vocab

    #reference code from : https://gist.github.com/amnrzv/596ba910524e0b1b4e8fa2167fd773bf#file-a_language_analysis-py-L2
    def lemmatize_words(self, pos_tagged_array):
        """
        Convert each word in the input to its base form
        and save it in a list
        """
        base_words = []
        for tag in pos_tagged_array:
            base_word = wordnet_lemmatizer.lemmatize(tag[0].lower(), 'v')
            base_words.append(base_word)
        return base_words

    def hot_one_encode_labels(self, data):
        self.labels_words = np.unique(np.asarray(data[1]))
        self.train_hot_one_labels = np.zeros(len(data[1]))
        for i in range(len(data[1])):
            self.train_hot_one_labels[i] = int(np.where(self.labels_words == data[1][i])[0])

    def generate_bag_of_words(self, filename, data):
        data_array = np.asarray(data).transpose()
        labels = np.unique(data_array[:,-1])

        for i in range(len(labels)):
            index_class = np.where(data_array[:,-1] == labels[i])
            class_data = data_array[index_class[0], :-1]
            self.generate_bag_of_words_class(class_data, filename + "_class_"+ str(i), self.train_hot_one_labels[index_class[0]])
        
    def generate_bag_of_words_class(self, data, filename, hot_one_labels):
        bag = np.zeros((len(data), len(self.vocab)+1))
        bag[:,-1] = hot_one_labels
        for i in range(data.shape[0]):
            sentence = data[i,:][0]
            if sentence.strip():
                words = word_tokenize(sentence)
                for tag in nltk.pos_tag(words):
                    # eliminating unnecessary POS tags
                    match = re.search('\w.*', tag[1])
                    if match:
                        base_word = wordnet_lemmatizer.lemmatize(tag[0].lower(), 'v')
                        index = self.get_vocab_index(base_word)
                        if index != -1:
                            bag[i,index] += 1
        self.save_pickle_file(bag, filename)

    def get_vocab_index(self, word):
        for i in range(len(self.vocab)):
            if self.vocab[i] == word:
                return i
        return -1

    def save_pickle_file(self, vocab, filename):
        with open(filename + '.pkl', 'wb') as f:
            pkl.dump(vocab, f)

    def load_pickle_file(self, filename):
        with open(filename + '.pkl', 'rb') as f:
            return np.asarray(pkl.load(f))
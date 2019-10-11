import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

class NaiveBayesClassifer():
    def __init__(self, train_data, vocabulary):
        self.train_inputs = train_data[:,:-1]
        self.labels = np.unique(train_data[:, -1])
        self.vocab = vocabulary

    def generate_bag_of_words(self, data):
        bag = np.zeros((1, len(self.vocab)))
        word_count = 0
        for i in range(data.shape[0]):
            #print("calculating {0}/{1}".format(i,data.shape[0]))
            sentence = data[i,:][0]
            for word in word_tokenize(sentence):
                base_word = wordnet_lemmatizer.lemmatize(word.lower(), 'v')
                #if base_word in self.vocab or word in self.vocab:
                index = self.get_vocab_index(base_word)
                if index != -1:
                    print("update word")
                    bag[0,index] += 1
                    word_count += 1
        return bag, word_count

    def train(self, data):
        print("started training")
        num_documents = data.shape[0]
        self.prob_classes = {}
        self.num_words_class = {}
        self.priors = {}
        for i in range(len(self.labels)):
            index_class = np.where(data[:,-1] == self.labels[i])
            class_data = data[index_class[0], :-1]
            print("Generating bag train data for {}".format(self.labels[i]))
            bag_class, word_count = self.generate_bag_of_words(class_data)
            self.num_words_class[str(self.labels[i])] = word_count
            self.prob_classes[str(self.labels[i])] = list(bag_class)
            self.priors[str(self.labels[i])] = class_data.shape[0]/num_documents
            print("Done calculatin prob for class : {}".format(self.labels[i]))

    def test_accuracy(self, test):
        prob_c = np.zeros(len(self.labels))
        prediction = []
        for s in test:
            words = word_tokenize(s[0])
            word_count = 0
            for i in range(len(self.labels)):
                prob_words = 0
                for word in words:
                    word_count += 1
                    base_word = wordnet_lemmatizer.lemmatize(word.lower(), 'v')
                    prob_word = np.log(1)
                    if base_word in self.vocab:
                        index = self.get_vocab_index(base_word)
                        prob_word = np.log(self.prob_classes[self.labels[i]][0][index] + 1)
                    prob_words += prob_word
                prob_c[i] = (prob_words - word_count*np.log(self.num_words_class[self.labels[i]] + len(self.vocab))) + np.log(self.priors[self.labels[i]])
            prediction.append(self.labels[np.argmax(np.asarray(prob_c))] == s[1])

        return np.sum(prediction)/test.shape[0]

    def label_encoding(self, labels):
        self.train_hot_one_labels = {}
        for i in range(len(self.labels)):
            self.train_hot_one_labels.update({self.labels[i], i})

    def get_vocab_index(self, word):
        for i in range(len(self.vocab)):
            if self.vocab[i] == word:
                return i
        return -1
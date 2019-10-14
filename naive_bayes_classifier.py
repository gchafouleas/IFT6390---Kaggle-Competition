import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl

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
            print("calculating {0}/{1}".format(i,data.shape[0]))
            sentence = data[i,:][0].split()
            for word in sentence:
                #if base_word in self.vocab or word in self.vocab:
                index = self.get_vocab_index(word)
                if index != -1:
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
            print("Label {0}/{1}".format(i+1, len(self.labels)))
            index_class = np.where(data[:,-1] == self.labels[i])
            class_data = data[index_class[0], :-1]
            print("Generating bag train data for {}".format(self.labels[i]))
            bag_class, word_count = self.generate_bag_of_words(class_data)
            self.num_words_class[str(self.labels[i])] = word_count
            self.prob_classes[str(self.labels[i])] = list(bag_class)
            self.priors[str(self.labels[i])] = class_data.shape[0]/num_documents
            print("Done calculating prob for class : {}".format(self.labels[i]))
        with open('prob_classes' + '.pkl', 'wb') as f:
            pkl.dump(self.prob_classes, f)
        with open('priors' + '.pkl', 'wb') as f:
            pkl.dump(self.priors, f)
        with open('num_words_class' + '.pkl', 'wb') as f:
            pkl.dump(self.num_words_class, f)

    def test_accuracy(self, test):
        prob_c = np.zeros(len(self.labels))
        prediction = []
        test_item = 0
        total_test_item = len(test)
        for s in test:
            test_item += 1
            word_count = 0
            words = s.split()
            print("test exaple {0}/{1}".format(test_item,total_test_item))
            for i in range(len(self.labels)):
                print("Computing labels for test")
                prob_words = 0
                for word in words:
                    word_count += 1
                    prob_word = np.log(1)
                    if word in self.vocab:
                        index = self.get_vocab_index(word)
                        prob_word = np.log(self.prob_classes[self.labels[i]][0][index] + 1)
                    prob_words += prob_word
                prob_c[i] = (prob_words - word_count*np.log(self.num_words_class[self.labels[i]] + len(self.vocab))) + np.log(self.priors[self.labels[i]])
            prediction.append(self.labels[np.argmax(np.asarray(prob_c))])

        return prediction

    def label_encoding(self, labels):
        self.train_hot_one_labels = {}
        for i in range(len(self.labels)):
            self.train_hot_one_labels.update({self.labels[i], i})

    def get_vocab_index(self, word):
        for i in range(len(self.vocab)):
            if self.vocab[i] == word:
                return i
        return -1
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl

wordnet_lemmatizer = WordNetLemmatizer()

class NaiveBayesClassifer():
    def __init__(self, train_data, vocabulary):
        self.labels = np.unique(train_data[:, -1])
        self.vocab = vocabulary

    def generate_bag_of_words(self, data):
        bag = dict(zip(self.vocab, [0]*len(self.vocab)))
        word_count = 0
        for i in range(data.shape[0]):
            print("calculating {0}/{1}".format(i,data.shape[0]))
            sentence = data[i,:][0].split()
            for word in sentence:
                #if base_word in self.vocab or word in self.vocab:
                if word in bag.keys():
                    bag[str(word)] += 1
                    word_count += 1

        return bag, word_count

    def train(self, data):
        print("started training")
        num_documents = data.shape[0]
        self.prob_classes = []
        self.num_words_class = {}
        self.priors = {}
        for i in range(len(self.labels)):
            print("Label {0}/{1}".format(i+1, len(self.labels)))
            index_class = np.where(data[:,-1] == self.labels[i])
            class_data = data[index_class[0], :-1]
            print("Generating bag train data for {}".format(self.labels[i]))
            bag_class, word_count = self.generate_bag_of_words(class_data)
            self.num_words_class[str(self.labels[i])] = word_count
            self.prob_classes.append(bag_class)
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
            words = s.split()
            print("test exaple {0}/{1}".format(test_item,total_test_item))
            for i in range(len(self.labels)):
                prob_words = 0
                word_count = 0
                oov = 0
                for word in words:
                    word_count += 1
                    if word in self.prob_classes[i].keys():
                    	oov += 1
                    prob_word = self.prob_classes[i].get(word,0)
                    prob = np.log(prob_word  + 1)
                    prob_words += prob
                prob_words -= word_count*np.log(self.num_words_class[self.labels[i]] + len(self.vocab) + oov)
                prob_c[i] = prob_words + np.log(self.priors[self.labels[i]])
            pred = self.labels[np.argmax(np.asarray(prob_c))]
            prediction.append(pred)

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

    def load_train_data(self):
        self.prob_classes = np.load("prob_classes.pkl", allow_pickle=True)
        #temp = []
        #for key, values in self.prob_classes.items():
        #    dict = {}
        #    for i in range(len(self.vocab)):
        #        dict[str(self.vocab[i])] = values[0][i]
        #    temp.append(dict)
        self.prob_classes = temp
        self.priors = np.load("priors.pkl", allow_pickle=True)
        self.num_words_class = np.load("num_words_class.pkl", allow_pickle=True)
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl

class NaiveBayesClassifer():
    def __init__(self, train_data, vocabulary):
        self.labels = np.unique(train_data[:, -1])
        self.vocab = np.unique(vocabulary)

    def generate_bag_of_words(self, data):
        bag = dict(zip(self.vocab, [0]*len(self.vocab)))
        word_count = 0
        for i in range(data.shape[0]):
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

    def test_accuracy(self, test, validation = False, alpha = 1):
        prob_c = np.zeros(len(self.labels))
        prediction = []
        test_item = 0
        total_test_item = len(test)
        for s in test:
            test_item += 1
            if validation:
            	words = s[0].split()
            else: 
            	words = s.split()
            #print("test exaple {0}/{1}".format(test_item,total_test_item))
            for i in range(len(self.labels)):
                prob_words = 0
                word_count = 0
                oov = 0
                for word in words:
                    word_count += 1
                    if word in self.prob_classes[i].keys():
                    	oov += 1
                    prob_word = self.prob_classes[i].get(word,0)
                    prob = np.log(prob_word  + alpha)
                    prob_words += prob
                prob_words -= word_count*np.log(self.num_words_class[self.labels[i]] + alpha*(len(self.vocab)))
                prob_c[i] = prob_words + np.log(self.priors[self.labels[i]])
            pred = self.labels[np.argmax(np.asarray(prob_c))]
            prediction.append(pred)

        return prediction
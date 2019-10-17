
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl


# TEST 2 GIT

wordnet_lemmatizer = WordNetLemmatizer()

class PreprocessData:
    def __init__(self):
        pass

    def generate_vocabulary(self, data):
        print("Generating vocabulary")
        text = self.process_input(data[:,:-1])
        print("preprocessed")
        vocab = self.lemmatize_words(text)
        print("lemma done")
        self.save_pickle_file(vocab,'vocab')
        print("Vocabulary saved")
        return vocab

    def normalize_text(self, data, file_name, train=True):
        normalize_text = data         
        if train: 
            data = data[:,:-1]
        for i in range(data.shape[0]):
            words = ""
            if train:
                sentence = data[i][0]
            else:
                sentence = data[i]
            for word in word_tokenize(sentence):
                base_word = wordnet_lemmatizer.lemmatize(word.lower(), 'v')
                #if base_word in self.vocab or word in self.vocab:
                words += " " + base_word
            if train:
                normalize_text[i,:-1] = words
            else:
                normalize_text[i] = words
        #self.save_pickle_file(normalize_text, file_name)
        return normalize_text
                
    #SOURCE code from : https://gist.github.com/amnrzv/596ba910524e0b1b4e8fa2167fd773bf#file-a_language_analysis-py-L2
    def process_input(self, data):
        vocab = []
        for sentence in data:
            words = word_tokenize(sentence[0])
            for tag in nltk.pos_tag(words):
                # eliminating unnecessary POS tags
                match = re.search('\w.*', tag[1])
                if match:
                    vocab.append(tag)
        return vocab

    #SOURCE code from : https://gist.github.com/amnrzv/596ba910524e0b1b4e8fa2167fd773bf#file-a_language_analysis-py-L2
    def lemmatize_words(self, pos_tagged_array):
        """
        Convert each word in the input to its base form
        and save it in a list
        """
        base_words = []
        for tag in pos_tagged_array:
            base_word = wordnet_lemmatizer.lemmatize(tag[0].lower(), 'v')
            if base_word not in base_words:
                base_words.append(base_word)
        return base_words      

    def save_pickle_file(self, vocab, filename):
        with open(filename + '.pkl', 'wb') as f:
            pkl.dump(vocab, f)

    def load_pickle_file(self, filename):
        with open(filename + '.pkl', 'rb') as f:
            return np.asarray(pkl.load(f))
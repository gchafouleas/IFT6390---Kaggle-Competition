
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

    def generate_vocabulary(self, data):
        print("Generating vocabulary")
        vocab = self.process_input(data[:,0])
        vocab = self.lemmatize_words(vocab)
        #self.save_pickle_file(vocab,'vocab')
        return vocab
        print("Vocabulary saved")

    #SOURCE code from : https://gist.github.com/amnrzv/596ba910524e0b1b4e8fa2167fd773bf#file-a_language_analysis-py-L2
    def process_input(self, data):
        vocab = []
        for sentence in data:
            words = word_tokenize(sentence)
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
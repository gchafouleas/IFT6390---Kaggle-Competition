
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl
from nltk.corpus import stopwords


# TEST 2 GIT


# TEST 2 GIT


# TEST 2 GIT

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class PreprocessData:
    def __init__(self):
        pass

    def generate_vocabulary(self, data):
        print("Generating vocabulary")
        vocab = self.process_input(data[:,:-1])
        print("preprocessed")
        vocab = self.lemmatize_words(vocab)
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
                if word not in stop_words:
                    word = wordnet_lemmatizer.lemmatize(word.lower(), 'v')
                    words += " " + word.lower()
            if train:
                normalize_text[i,:-1] = words
            else:
                normalize_text[i] = words

        return normalize_text
                
    #SOURCE code from : https://gist.github.com/amnrzv/596ba910524e0b1b4e8fa2167fd773bf#file-a_language_analysis-py-L2
    def process_input(self, data):
        vocab = []
        for sentence in data:
            words = word_tokenize(sentence[0])
            for tag in words:
                if tag not in stop_words:
                    if tag not in vocab:
                        vocab.append(tag.lower())
        return vocab

    #SOURCE code from : https://gist.github.com/amnrzv/596ba910524e0b1b4e8fa2167fd773bf#file-a_language_analysis-py-L2
    def lemmatize_words(self, pos_tagged_array):
        """
        Convert each word in the input to its base form
        and save it in a list
        """
        base_words = []
        for tag in pos_tagged_array:
            base_word = wordnet_lemmatizer.lemmatize(tag.lower(), 'v')
            base_words.append(base_word)
        return base_words      

    def save_pickle_file(self, vocab, filename):
        with open(filename + '.pkl', 'wb') as f:
            pkl.dump(vocab, f)

    def load_pickle_file(self, filename):
        with open(filename + '.pkl', 'rb') as f:
            return np.asarray(pkl.load(f))
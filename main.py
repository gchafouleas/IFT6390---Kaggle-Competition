import numpy as np
from preproces_data import PreprocessData
from naive_bayes_classifier import NaiveBayesClassifer
import pandas as pd

train_data = np.load("./data/data_train.pkl", allow_pickle=True)
test_data = np.load("./data/data_test.pkl", allow_pickle=True)
train_data = np.asarray(train_data).transpose()
test_data = np.asarray(test_data).transpose()

preprocess = PreprocessData()
vocab = preprocess.load_pickle_file('vocab')
print("Normalizing train data")
train_data = preprocess.normalize_text(train_data, 'train_data_normalized')
print("Normalizing test data")
test_data = preprocess.normalize_text(test_data, 'test_data_normalized', train=False)
naive_classifier = NaiveBayesClassifer(train_data, vocab)
naive_classifier.train(train_data)
predictions = naive_classifier.test_accuracy(test_data)
print(predictions)
df = pd.read_csv('./data/sample_submission.csv', delimiter= ',')
df['Category'] = predictions
df.to_csv('./prediction.csv', index=False, sep=',')   
print("saved prediction csv")

# TEST GIT SUBMIT
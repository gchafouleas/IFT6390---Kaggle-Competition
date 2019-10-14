import numpy as np
#import cupy as np
import torch
from preproces_data import PreprocessData
from naive_bayes_classifier import NaiveBayesClassifer
import pandas as pd
train_data = np.load("./data/data_train.pkl", allow_pickle=True)
test_data = np.load("./data/data_test.pkl", allow_pickle=True)
train_data = np.asarray(train_data).transpose()
test_data = np.asarray(test_data).transpose()
#np.random.shuffle(data)
#i_train = int(data.shape[0] * 0.7)
#i_test = int(data.shape[0] - data.shape[0] * 0.3)
#train_data = train_data[:100,:]
#print(train_data.shape)
#test_data = data[2:, :]

preprocess = PreprocessData(train_data)
vocab = preprocess.load_pickle_file('vocab')
print("Normalizing train data")
train_data = preprocess.normalize_text(train_data, 'train_data_normalized')
print("Normalizing test data")
test_data = preprocess.normalize_text(test_data, 'test_data_normalized', train=False)
naive_classifier = NaiveBayesClassifer(train_data, vocab)
naive_classifier.train(train_data)
#naive_classifier.load_train_data()
predictions = naive_classifier.test_accuracy(test_data)
df = pd.read_csv('./data/sample_submission.csv', delimiter= ',')
df['Category'] = predictions
df.to_csv('./prediction.csv', index=False, sep=',')   
print("saved prediction csv")
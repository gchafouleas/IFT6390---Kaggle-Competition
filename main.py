import numpy as np
from preproces_data import PreprocessData
from naive_bayes_classifier import NaiveBayesClassifer

data = np.load("./data/data_train.pkl", allow_pickle=True)
data = np.asarray(data).transpose()
np.random.shuffle(data)
i_train = int(data.shape[0] * 0.7)
i_test = int(data.shape[0] - data.shape[0] * 0.3)
train_data = data[:i_train,:]
print(train_data.shape)
test_data = data[i_test:, :]
print(test_data.shape)

preprocess = PreprocessData(train_data)
vocab = preprocess.generate_vocabulary(train_data)
print(len(vocab))
naive_classifier = NaiveBayesClassifer(train_data, vocab)
naive_classifier.train(train_data)
accuracy = naive_classifier.test_accuracy(test_data)
print(accuracy)
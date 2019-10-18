import numpy as np
from preproces_data import PreprocessData
from naive_bayes_classifier import NaiveBayesClassifer
import pandas as pd

print("start")
np.random.seed(3395)
train_data = np.load("./data/data_train.pkl", allow_pickle=True)
test_data = np.load("./data/data_test.pkl", allow_pickle=True)
train_data = np.asarray(train_data).transpose()
test_data = np.asarray(test_data).transpose()
labels = np.unique(train_data[:, -1])
class_index = []
for l in labels:
	class_index.append(np.where(train_data[:,-1] == l)[0])

train_class = []
test_class = []
for i in class_index:
	np.random.shuffle(i)
	train_class.append(train_data[i[:2500]])
	test_class.append(train_data[i[2500:]])
train_data_test = np.concatenate(train_class)
validation_data = np.concatenate(test_class)

preprocess = PreprocessData()
#vocab = preprocess.load_pickle_file('vocab')
vocab = preprocess.generate_vocabulary(train_data_test)
print(len(np.unique(vocab)))
print("Normalizing train data")
train_data = preprocess.normalize_text(train_data, 'train_data_normalized')
train_data_test = preprocess.normalize_text(train_data_test, 'train_data_normalized')
print("Normalizing test data")
validation_data = preprocess.normalize_text(validation_data, 'validation_data_normalized')
test_data = preprocess.normalize_text(test_data, 'test_data_normalized', train=False)
naive_classifier = NaiveBayesClassifer(train_data_test, vocab)
naive_classifier.train(train_data_test)
alphas = np.linspace(0.09,0.2, 15)
accuracies = []
for alpha in alphas:
	preds = naive_classifier.test_accuracy(validation_data[:,:-1], True, alpha)
	accuracy = np.mean(preds == validation_data[:,-1])
	accuracies.append(accuracy)
	print("accuracy {0} for {1}".format(accuracy, str(alpha)))
print("Best alpha: ", alphas[np.argmax(accuracies)])
#vocab = preprocess.generate_vocabulary(train_data)
#naive_classifier_test = NaiveBayesClassifer(train_data, vocab)
#naive_classifier_test.train(train_data)
predictions = naive_classifier.test_accuracy(test_data, False, alphas[np.argmax(accuracies)])
df = pd.read_csv('./data/sample_submission.csv', delimiter= ',')
df['Category'] = predictions
df.to_csv('./prediction.csv', index=False, sep=',')   
print("saved prediction csv")

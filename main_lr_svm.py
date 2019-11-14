from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from preproces_data import PreprocessData
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import pickle
from sklearn.metrics import precision_recall_fscore_support
import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier

#Source : https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
#Source: https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0

def save_precision(filename, items, accuracy_valid):
    print("saving {}".format(filename))
    with open(filename, 'a') as fin:
        fin.write("precision: " + str(items[0])+ "\n") 
        fin.write("recall: " + str(items[1])+ "\n") 
        fin.write("f1score: " + str(items[2])+ "\n") 
        fin.write("support: " + str(items[3])+ "\n") 
        fin.write("Accuracy validation: " + str(accuracy_valid) + "\n")

def save_model(file_name, model):
    print("Saving {}".format(file_name))
    with open(file_name, 'wb') as fin:
        pickle.dump(model, fin)

def svc_param_selection(X, y, nfolds):
    Cs = np.linspace(0.001, 10, num=5)
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(svm.LinearSVC(penalty='l2'), param_grid, cv=nfolds, verbose=4)
    grid_search.fit(X, y)
    return grid_search

def LR_param_selection(X, y, nfolds):
    Cs = np.linspace(3, 10, num=5)
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(LogisticRegression(solver='newton-cg', multi_class='multinomial'), param_grid, cv=nfolds, verbose=4)
    grid_search.fit(X, y)
    return grid_search

rs = np.random.seed(500)
train = np.load("./data/data_train.pkl", allow_pickle=True)
train = np.asarray(train).transpose()
preprocess = PreprocessData()
print("Normalizing data")
#train = preprocess.normalize_text(train, 'train_data_normalized')

train_data, validation_data, train_labels, valid_labels = model_selection.train_test_split(train[:,:-1].ravel(),train[:,-1],test_size=0.1)

Encoder = LabelEncoder()
train_labels = Encoder.fit_transform(train_labels)
valid_labels = Encoder.fit_transform(valid_labels)

print("Vectorizing data")
Tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
Tfidf_vect.fit(train[:,:-1].ravel())
with open('models/train_vectorizer_n_grams.pk', 'wb') as fin:
    pickle.dump(Tfidf_vect, fin)

Train_X_Tfidf = Tfidf_vect.transform(train_data)
Test_X_Tfidf = Tfidf_vect.transform(validation_data)

print("training logistic regression classifier... Time is ", datetime.datetime.now().time())
logreg = LR_param_selection(Train_X_Tfidf, train_labels, 3)
print("Done training Logistic regression. Time is ", datetime.datetime.now().time(), "\n")
predictions_LR = logreg.predict(Test_X_Tfidf)
save_model('models/logistic_regression_model_n_grams.pkl', logreg)
precision_LR = precision_recall_fscore_support(valid_labels, predictions_LR, average='macro')
accuracy_LR = accuracy_score(predictions_LR, valid_labels)*100
print("Best parameters: ", logreg.best_params_)
print("LR Accuracy Score -> valid: {0}".format(accuracy_LR))
save_precision('models/logistic_regression_precision_n_grams.txt', precision_LR, accuracy_LR)
print("Classification report LR: ")
print(classification_report(valid_labels, predictions_LR))

print("training SVM classifier... Time is ", datetime.datetime.now().time())
SVM = svc_param_selection(Train_X_Tfidf, train_labels, 3)
print("Done training SVM. Time is ", datetime.datetime.now().time(), "\n")
predictions_SVM = SVM.predict(Test_X_Tfidf)
save_model('models/svm_model_n_grams.pkl', SVM)
precision_SVM = precision_recall_fscore_support(valid_labels, predictions_SVM, average='macro')
# Use accuracy_score function to get the accuracy
accuracy_SVM = accuracy_score(predictions_SVM, valid_labels)*100
print("Best parameters: ", SVM.best_params_)
print("SVM Accuracy Score -> valid: {0}".format(accuracy_SVM))
save_precision('models/svm_precision_n_grams.txt', precision_SVM, accuracy_SVM)
print("Classification report SVM: ")
print(classification_report(valid_labels, predictions_SVM))
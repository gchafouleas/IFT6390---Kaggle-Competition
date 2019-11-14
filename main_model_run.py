from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
import pandas as pd


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

def ridgide_param_selection(X, y, nfolds):
    alphas = np.linspace(0.01, 10, num=20)
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(RidgeClassifier(solver='sag'), param_grid, cv=nfolds, verbose=4)
    grid_search.fit(X, y)
    return grid_search

def NB_param_selection(X, y, nfolds):
    alphas = np.linspace(0.001, 10.0, num=100)
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(naive_bayes.ComplementNB(), param_grid, cv=nfolds, verbose=4)
    grid_search.fit(X, y)
    return grid_search

rs = np.random.seed(500)
split_data = True
train = np.load("./data/data_train.pkl", allow_pickle=True)
train = np.asarray(train).transpose()
preprocess = PreprocessData()
#print("Normalizing data")
#train = preprocess.normalize_text(train, 'train_data_normalized')
if split_data:
    train_data, validation_data, train_labels, valid_labels = model_selection.train_test_split(train[:,:-1].ravel(),train[:,-1],test_size=0.1)
else:
    train_data = train[:,:-1].ravel()
    train_labels = train[:,-1]

Encoder = LabelEncoder()
train_labels = Encoder.fit_transform(train_labels)
if split_data:
    valid_labels = Encoder.fit_transform(valid_labels)

print("Vectorizing data")
Tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
Tfidf_vect.fit(train[:,:-1].ravel())
with open('models/train_vectorizer_n_grams_normalized.pk', 'wb') as fin:
    pickle.dump(Tfidf_vect, fin)

Train_X_Tfidf = Tfidf_vect.transform(train_data)
if split_data:
    Test_X_Tfidf = Tfidf_vect.transform(validation_data)


print("training Naive Bayes classifier... Time is ", datetime.datetime.now().time())
NB = NB_param_selection(Train_X_Tfidf, train_labels, 3)
print("Done training Naive Bayes. Time is ", datetime.datetime.now().time(), "\n")
save_model('models/Complement_Naive_bayes_model.pkl', NB)
if split_data:
    predictions_NB = NB.predict(Test_X_Tfidf)
    precision_NB = precision_recall_fscore_support(valid_labels, predictions_NB, average='macro')
    accuracy_NB = accuracy_score(predictions_NB, valid_labels)*100
    print("Com Accuracy Score -> valid: {0}".format(accuracy_NB))
    save_precision('models/Complement_Naive_bayes_precision.txt', precision_NB, accuracy_NB)
    print("Classification report LR: ")
    print(classification_report(valid_labels, predictions_NB))
print("Best parameters: ", NB.best_params_)

print("training Rigide Regression classifier... Time is ", datetime.datetime.now().time())
ridgide = ridgide_param_selection(Train_X_Tfidf, train_labels, 3)
print("Done training Rigide Regression. Time is ", datetime.datetime.now().time(), "\n")
predictions_R = ridgide.predict(Test_X_Tfidf)
save_model('models/Rigide_Regression_model.pkl', ridgide)
if split_data:
    precision_R = precision_recall_fscore_support(valid_labels, predictions_R, average='macro')
    accuracy_R = accuracy_score(predictions_R, valid_labels)*100
    print("Com Accuracy Score -> valid: {0}".format(accuracy_R))
    save_precision('models/Rigide_Regression_precision.txt', precision_R, accuracy_R)
    print("Classification report LR: ")
    print(classification_report(valid_labels, predictions_R))
print("Best parameters: ", ridgide.best_params_)

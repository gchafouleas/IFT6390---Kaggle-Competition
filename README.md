# Kaggle 
This is the repository for the Kaggle competition code for IFT6390
## Installation
Before running the program you must first install all the requirements using the requirements.txt file. 
```
pip install -r requirement.txt
```
### Note:
We use the nltk library and aditional installation are required in order for the program to run. 
You will need to manually install some dependences from nltk by running your python interpreter as an example:
```
import nltk
nltk.download('punkt')
```

## Running Naive Bayes with smoothing
To run the Naive bayes classifier you simply have to run: 
```
python main.py
```

The main.py contains the code to train and test the Naive bayes with the loaded data. 

The naive_bayes_classifier.py is the class containing the code of the naive_bayes_classifier. 

The preproces_data.py is the class containing the preprocessing done to generate the vocabulary and normalize the data. 

It is configured to run spliting the train into a train an valid set and tuning on a range of alpha values for the hyperparamters. Picking the best alpha and test set with the best tuned alpha. 

## Running Logistic Regression and Linear SVM 
To run both the logistic regression and Linear SVM you simply need to run: 
```
python main_lr_svm.py
```
This will evaluate a both a logistic regression and Linear svm with GridSearchCV cv=3. 
The following parameters can be changed in the script to tune: 
TfidfVectorizer: you can play with the values of it in the script to change the results of your model. 
svc_param_selection: function that determines what hyperparameter it will search and also nfolds of the cross validation for the Linear SVM. 
LR_param_selection: function that determines what hyperparameter it will search and also nfolds of the cross validation for the Logistic Regression. 

The script also saves the vectorizer, model precisions and model pickle for the best run. 

## Running best models
This script run a Complement Naive Bayes and a Ridge Regression mode. You can run these models the using the following: 
```
python main_model_run.py
```
It is currently set up to run both the Complement Naive Bayes and Ridge Regression using GridSearchCV cv=3 and TfidfVectorizer(ngram_range=(1,2))
You can modify the following in the script to change the values generated. 
TfidfVectorizer: you can play with the values of it in the script to change the results of your model. 
NB_param_selection: function that determines what hyperparameter it will search and also nfolds of the cross validation for the Complement Naive Bayes.
ridgide_param_selection: function that determines what hyperparameter it will search and also nfolds of the cross validation for the Ridge Regression.
The script also saves the vectorizer, model precisions and model pickle for the best run.

## Evaluating the model 
In order to evaluate the model on the test set you can run: 
```
python evaluate_model.py
```
Two parameters are possible to change to set the location of the model_path and vectorizer_loc to use for evaluation. This will create a predictions.csv with the predicted classes. 

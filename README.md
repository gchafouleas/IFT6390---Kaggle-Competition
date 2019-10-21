# Kaggle Naive bayes with smoothing 
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

## Running
To run the Naive bayes classifier you simply have to run: 
```
python main.py
```

The main.py contains the code to train and test the Naive bayes with the loaded data. 

The naive_bayes_classifier.py is the class containing the code of the naive_bayes_classifier. 

The preproces_data.py is the class containing the preprocessing done to generate the vocabulary and normalize the data. 

It is configured to run spliting the train into a train an valid set and tuning on a range of alpha values for the hyperparamters. Picking the best alpha and test set with the best tuned alpha. 

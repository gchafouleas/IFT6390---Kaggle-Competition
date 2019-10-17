# Kaggle Naive bayes with smoothing 
This is the repository for the Kaggle competition code for IFT6390
##Installation
Before running the program you must first install all the requirements using the requirements.txt file. 
```
pip install -r requirement.txt
```
#Note:
We use the nltk library and aditional installation are required in order for the program to run. 
You will need to manually install some dependences from nltk by running your python interpreter as an example:
```
import nltk
nltk.download('punkt')
```

##Running
To run the Naive bayes classifier you simply have to run: 
```
python main.py
```
It is configured to run spliting the train into a train an valid set and tuning on a range of alpha values for the hyperparamters. Picking the best alpha and training again on the whole train set then evaluation the test set with the best tuned alpha. 

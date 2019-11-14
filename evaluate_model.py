import pickle 
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description='model ')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--model_path', type=str, default='models/Complement_Naive_bayes_model.pkl',
                    help='description of the type of test done on normalization data')
parser.add_argument('--vectorizer_loc', type=str, default='models/train_vectorizer_n_grams_normalized.pk',
                    help='location of the vectorizer')

args = parser.parse_args()
argsdict = args.__dict__

np.random.seed(500)
print("loading data")
train = np.load("./data/data_train.pkl", allow_pickle=True)
train = np.asarray(train).transpose()
test_data = np.load("./data/data_test.pkl", allow_pickle=True)
test_data = np.asarray(test_data).transpose()
print("done loading data")
with open(args.vectorizer_loc, 'rb') as f:
    Tfidf_vect = pkl.load(f)

Encoder = LabelEncoder()
train_labels = Encoder.fit_transform(train[:,-1])
test_final_input = Tfidf_vect.transform(test_data)

loaded_model = pickle.load(open(args.model_path, 'rb'))
predictions = loaded_model.predict(test_final_input)
df = pd.read_csv('./data/sample_submission.csv', delimiter= ',')
df['Category'] = Encoder.inverse_transform(predictions)
df.to_csv('./prediction.csv', index=False, sep=',')   
print("saved prediction csv")

import numpy as np
from preproces_data import PreprocessData

train_data = np.load("./data/data_train.pkl", allow_pickle=True)
#train_data_array = np.asarray(train_data).transpose()

preprocess = PreprocessData(train_data)
preprocess.generate_bag_of_words("train_data_bag", train_data)
#preprocess.generate_vocabulary(train_data)
#print(preprocess.train_data_bag[0,:])
train_class_0 = preprocess.load_pickle_file('train_data_bag_class_0')
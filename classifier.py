import random

from keras.layers import Input

from Data import Data
from Models import *
from WordVectors import WordVectors

random.seed(11)

data_fpath = '/home/ubuntu/datasets/hate_speech_davidson.csv'
glove_fpath = '/home/ubuntu/datasets/word_embeddings/glove.twitter.27B.200d.txt'

# load data from csv
data = Data(data_fpath)

# load glove word vector and subset with csv data
word_vectors = WordVectors(glove_fpath, data.word_index)

# instantiate SimpleCNN model object
model = SimpleCNN(data, word_vectors)

# train model
model.fit(data)

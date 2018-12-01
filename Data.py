import csv
import math
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class Data:
    def __init__(self, path):
        self.path = path

        # load sentences and labels from CSV
        self.sentences = []
        self.labels = []
        self._init_load()

        # convert labels into keras required format
        labels = np.asarray(self.labels)
        self.labels = to_categorical(labels)  # one-hot encoding

        # tokenize sentences into arrays of words (as integers)
        self.word_index = None
        self.sequences = None
        self._init_tokenize(self.sentences)

        # pad tokenized sentences
        self.max_seq_len = None
        self.padded_sequences = None
        self._init_pad(self.sequences)

        # split sequences into different sets of numpy arrays
        self.train_y = None 
        self.train_x = None
        self.valid_y = None
        self.valid_x = None
        self.test_y = None
        self.test_x = None
        self._init_split()


    def _init_load(self):
        with open(self.path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                self.sentences.append(row[-1])
                self.labels.append(int(row[-2]))

    def _init_tokenize(self, sentences):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        self.word_index = tokenizer.word_index
        self.sequences = tokenizer.texts_to_sequences(sentences)

    def _init_pad(self, sequences):
        max_seq_len = max([len(s) for s in sequences])
        self.max_seq_len = max_seq_len
        self.padded_sequences = pad_sequences(sequences, maxlen=max_seq_len)

    def _init_split(self):
        indices = np.arange(self.padded_sequences.shape[0])
        np.random.shuffle(indices)
        split_one = math.floor(len(indices) * .7)
        split_two = math.floor(len(indices) * .9)

        training_indices = indices[:split_one]
        train_x = [self.padded_sequences[i] for i in training_indices]
        self.train_x = np.array(train_x)
        self.train_y = self.labels[training_indices]

        validation_indices = indices[split_one:split_two]
        valid_x = [self.padded_sequences[i] for i in validation_indices]
        self.valid_x = np.array(valid_x)
        self.valid_y = self.labels[validation_indices]

        test_indices = indices[split_two:]
        test_x = [self.padded_sequences[i] for i in test_indices]
        self.test_x = np.array(test_x)
        self.test_y = self.labels[test_indices]

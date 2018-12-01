import numpy as np

from keras.layers import Embedding

class WordVectors:
    def __init__(self, path, word_index):
        self.path = path
        self.nrows = len(word_index)  # number of words
        
        # load glove file into dict {word: vectors<arr>}
        self.embeddings_index = {}
        self.ncols = None  # updated with number of dimensions
        self._load_file()

        # build matrix(m,n) where m == word_index and n == vectors(m)
        self.matrix = None
        self._build_matrix(word_index)

    def _build_matrix(self, word_index):
        self.matrix = np.zeros((self.nrows + 1, self.ncols))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.matrix[i] = embedding_vector

    def _load_file(self):
        # stream file and process
        f = open(self.path)
        line = next(f)
        self.ncols = len(line.split()) - 1
        self._process_line(line)
        for line in f:
            self._process_line(line)
        f.close()


    def _process_line(self, line):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        self.embeddings_index[word] = coefs


    def get_keras_embedding(self, max_seq_len):
        return Embedding(self.nrows + 1,
                         self.ncols,
                         weights=[self.matrix],
                         input_length=max_seq_len,
                         trainable=False)

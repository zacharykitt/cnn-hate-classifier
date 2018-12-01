from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Concatenate, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
)
from keras.models import Model, Sequential

from Callbacks import Metrics

class Classifier:
    def __init__(self, data, word_vectors):
        self.data= data
        self.word_vectors = word_vectors
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError


class SimpleCNN(Classifier):
    ''' Implements the model suggested by Yoon Kim (2014) '''
    batch_size = 128
    epochs = 10   # default: 15
    filters = 20  # default: 100
    kernel_sizes = [3, 4, 5]
    model_output = 'weights.best.hdf5'

    def build_model(self):
        max_seq_len = self.data.max_seq_len
        sequence_input = Input(shape=(max_seq_len,), dtype='int32')

        # instantiate a keras Embedding layer object
        word_embedding = self.word_vectors.get_keras_embedding(max_seq_len)

        # layer 1: load input data into Embedding
        embedded_seq = word_embedding(sequence_input)

        # layer 2: get features from pooled CNNs
        features = []
        for size in self.kernel_sizes:
            temp_conv = Conv1D(self.filters,
                               size,
                               activation='relu')(embedded_seq)
            temp_pool = MaxPooling1D()(temp_conv)
            features.append(Flatten()(temp_pool))

        # layer 3: concatenation
        concatenated = Concatenate()(features)
        dropout = Dropout(.5)(concatenated)

        # layer 4: make predictions with dense layer
        preds = Dense(3, activation='softmax')(dropout)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam')
        return model

    def fit(self, data):
        metrics = Metrics()
        checkpoint = ModelCheckpoint(self.model_output,
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min')
        self.model.fit(data.train_x,
                       data.train_y,
                       validation_data=(data.valid_x, data.valid_y),
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       callbacks=[metrics, checkpoint])

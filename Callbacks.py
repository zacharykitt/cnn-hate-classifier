import numpy as np

from keras.callbacks import Callback
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score
)

class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        preds = np.asarray(self.model.predict(self.validation_data[0])).round()
        targets = self.validation_data[1]
        f1 = f1_score(targets, preds, average='micro')
        accuracy = accuracy_score(targets, preds)
        print(classification_report(targets, preds))
        print(confusion_matrix(targets.argmax(axis=1), preds.argmax(axis=1)))
        print('accuracy: ' + str(accuracy))

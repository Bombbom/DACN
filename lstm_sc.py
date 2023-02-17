from __future__ import print_function
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, LSTM, ReLU, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adamax
# from sklearn.utils import compute_class_weight
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import numpy as np


# Read dataset and split: test and train
vector_filename = r'pre_train.pkl'
data = pd.read_pickle(vector_filename)
vectors = np.stack(data.iloc[:, 0].values)
labels = data.iloc[:, 1].values


x_train, x_test, y_train, y_test = train_test_split(vectors, labels,train_size=0.8,random_state=1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

dropout = 0.2

lr = 0.002
batch_size = 64
epochs = 10
threshold = 0.5

adamax = Adamax(lr)
model = Sequential()
model.add(LSTM(300, input_shape=(vectors.shape[1], vectors.shape[2])))
model.add(ReLU())
model.add(Dropout(dropout))
model.add(Dense(300))
model.add(ReLU())
model.add(Dropout(dropout))
model.add(Dense(2, activation='softmax'))


model.compile(loss='binary_crossentropy', optimizer=adamax, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


# evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

_, accuracy_test = model.evaluate(x_test, y_test)
print(accuracy_test)
_, accuracy_train = model.evaluate(x_train, y_train)
print(accuracy_train)
values = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
predictions = (model.predict(x_test, batch_size=batch_size)).round()

tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
FPR=fp / (fp + tn)
print(FPR)
FNR= fn / (fn + tp)
print(FNR)
recall = tp / (tp + fn)
print(recall)
precision = tp / (tp + fp)
print(precision)
F1=(2 * precision * recall) / (precision + recall)
print(F1)
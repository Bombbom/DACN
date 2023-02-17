from __future__ import print_function
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout,  ReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adamax
import warnings
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

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

# Model
adamax = Adamax(lr)
model = Sequential()
model.add(SimpleRNN(300, input_shape=(vectors.shape[1], vectors.shape[2])))
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

predictions = (model.predict(x_test, batch_size=batch_size)).round()
predictions = (predictions >= threshold)
tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1)).ravel()

print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
print('False positive rate(FPR): ', fp / (fp + tn))
print('False negative rate(FNR): ', fn / (fn + tp))
recall = tp / (tp + fn)
print('Recall(TPR): ', recall)
precision = tp / (tp + fp)
print('Precision: ', precision)
print('F1 score: ', (2 * precision * recall) / (precision + recall))

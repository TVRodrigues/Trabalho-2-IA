import numpy as np 
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


Xt = pd.read_csv('HARDataset/train/X_train.txt', header=None, delim_whitespace=True)
Yt = pd.read_csv('HARDataset/train/y_train.txt', header=None, delim_whitespace=True)
Xts = pd.read_csv('HARDataset/test/y_test.txt', header=None, delim_whitespace=True)
Yts = pd.read_csv('HARDataset/test/y_test.txt', header=None, delim_whitespace=True)

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

clf.fit(Xt, Yt)
y_pred = clf.predict(Xts)

accuracy_score(Yts, y_pred)
cm = confusion_matrix(Yts, y_pred)
print(cm)

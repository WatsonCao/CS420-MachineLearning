#coding=utf-8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import numpy as np

mnist = fetch_mldata("MNIST original")
mnist.data,mnist.target = shuffle(mnist.data,mnist.target)
mnist.data = mnist.data[:50000]
mnist.target = mnist.target[:50000]
print('finished reading the data')
X_train,X_test,y_train,y_test = train_test_split(mnist.data,mnist.target,test_size=0.2,random_state=0)
model = SVC(kernel='linear')
model.fit(X_train,y_train)
print('finished training')
print('training accuracy:',model.score(X_train,y_train))
print('training accuracy:',model.score(X_test,y_test))


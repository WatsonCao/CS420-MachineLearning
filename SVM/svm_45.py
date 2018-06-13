#coding=utf-8
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import numpy as np
data_num = 60000 #The number of figures
fig_w = 45       #width of each figure

data = np.fromfile("D:\课程\机器学习\大作业\MNIST\mnist_train_data",dtype=np.uint8)
label = np.fromfile("D:\课程\机器学习\大作业\MNIST\mnist_train_label",dtype=np.uint8)
data = data.reshape(data_num,fig_w*fig_w)




test_num=10000
data_test = np.fromfile("D:\课程\机器学习\大作业\MNIST\mnist_test_data",dtype=np.uint8)
label_test = np.fromfile("D:\课程\机器学习\大作业\MNIST\mnist_test_label",dtype=np.uint8)
data_test = data_test.reshape(test_num,fig_w*fig_w)



model = SVC(kernel='poly')
model.fit(data[:10000],label[:10000])
print('finished training')

print('training accuracy:',model.score(data_test,label_test))


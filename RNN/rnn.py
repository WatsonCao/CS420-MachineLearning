# -*- coding: utf-8 -*-
# @Time    : 2018/6/3 0013 10:53
# @Author  : WatsonCao
# @FileName: rnn.py
# @Software: PyCharm Community Edition

#This file is used to trian on the given dataset

import tensorflow as tf
import numpy as np
import random

#The number of figures
data_num = 60000

#width of each figure
fig_w = 45

#read training data from given dataset and adjust the format of label
data_chj = np.fromfile("mnist_train_data",dtype=np.uint8)
label_chj = np.fromfile("mnist_train_label",dtype=np.uint8)
data_chj = data_chj.reshape(data_num,fig_w,fig_w)

zero=[0]
arr=zero*data_num*9
column=np.array(arr)
column=column.reshape(data_num,9)
label_chj=np.column_stack((label_chj,column))
for i in range(label_chj.shape[0]):
    pos=label_chj[i,0]
    label_chj[i,0]=0
    label_chj[i,pos]=1
    

#read testing data from given dataset and adjust the format of label
test_num=10000
data_test_chj = np.fromfile("mnist_test_data",dtype=np.uint8)
label_test_chj = np.fromfile("mnist_test_label",dtype=np.uint8)
data_test_chj = data_test_chj.reshape(test_num,fig_w,fig_w)

zero=[0]
arr=zero*test_num*9
column=np.array(arr)
column=column.reshape(test_num,9)
label_test_chj=np.column_stack((label_test_chj,column))
for i in range(label_test_chj.shape[0]):
    pos=label_test_chj[i,0]
    label_test_chj[i,0]=0
    label_test_chj[i,pos]=1

sess = tf.InteractiveSession()

#parameters of network
learning_rate = 0.00002
batch_size = 100

n_input = 45
n_steps = 45
n_hidden = 50
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

def RNN(x, n_steps, n_input, n_hidden, n_classes):
    # Parameters:
    # Input gate: input, previous output, and bias
    ix = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.0001, 0.0001))
    im = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.0001, 0.0001))
    ib = tf.Variable(tf.zeros([1, n_hidden]))
    # Forget gate: input, previous output, and bias
    fx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.0001, 0.0001))
    fm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.0001, 0.0001))
    fb = tf.Variable(tf.zeros([1, n_hidden]))
    # Memory cell: input, state, and bias
    cx = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.0001, 0.0001))
    cm = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.0001, 0.0001))
    cb = tf.Variable(tf.zeros([1, n_hidden]))
    # Output gate: input, previous output, and bias
    ox = tf.Variable(tf.truncated_normal([n_input, n_hidden], -0.0001, 0.0001))
    om = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], -0.0001, 0.0001))
    ob = tf.Variable(tf.zeros([1, n_hidden]))
    # Classifier weights and biases
    w = tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # Definition of the cell computation
    def lstm_cell(i, o, state):
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        update = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
        state = forget_gate * state + input_gate * update
        return output_gate * tf.tanh(state), state

    # Unrolled LSTM loop
    outputs = list()
    state = tf.Variable(tf.zeros([batch_size, n_hidden]))
    output = tf.Variable(tf.zeros([batch_size, n_hidden]))

    # x shape: (batch_size, n_steps, n_input)
    # desired shape: list of n_steps with element shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps, 0)
    for i in x:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)
    logits = tf.matmul(outputs[-1], w) + b
    return logits

pred = RNN(x, n_steps, n_input, n_hidden, n_classes)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
hhh=tf.argmax(pred, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


# Launch the graph
sess.run(init)
for step in range(40000):
    #randomly generate the training batch
    method=random.random()
    ran_pos=random.randint(1,100)
    if(method>0.5):
        batch_x  = data_chj[(step%599)*batch_size+ran_pos:(step%599+1)*batch_size+ran_pos,:]
        batch_y = label_chj[(step%599)*batch_size+ran_pos:(step%599+1)*batch_size+ran_pos,:]
    else:
        batch_x = data_chj[(step % 599+1) * batch_size - ran_pos:(step % 599 + 2) * batch_size - ran_pos, :]
        batch_y = label_chj[(step % 599+1) * batch_size - ran_pos:(step % 599 + 2) * batch_size - ran_pos, :]
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

    if step % 50 == 0:
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
print("Optimization Finished!")



# # Calculate accuracy for 100 mnist test images
test_len = batch_size
test_data = data_test_chj[:test_len,:]
test_label = label_test_chj[:test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[test_len:2*test_len,:]
test_label = label_test_chj[test_len:2*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[2*test_len:3*test_len,:]
test_label = label_test_chj[2*test_len:3*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[3*test_len:4*test_len,:]
test_label = label_test_chj[3*test_len:4*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[4*test_len:5*test_len,:]
test_label = label_test_chj[4*test_len:5*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[5*test_len:6*test_len,:]
test_label = label_test_chj[5*test_len:6*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[6*test_len:7*test_len,:]
test_label = label_test_chj[6*test_len:7*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[7*test_len:8*test_len,:]
test_label = label_test_chj[7*test_len:8*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[8*test_len:9*test_len,:]
test_label = label_test_chj[8*test_len:9*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
test_data = data_test_chj[9*test_len:10*test_len,:]
test_label = label_test_chj[9*test_len:10*test_len,:]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
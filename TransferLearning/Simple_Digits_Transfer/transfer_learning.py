# -*- coding: utf-8 -*-
# @Time    : 2018/6/2 0013 19:20
# @Author  : WatsonCao
# @FileName: transfer_learning.py
# @Software: PyCharm Community Edition


#First of all, we train a simple CNN on 1,4,5,9 digits, then use this model to trian on 0,2,3,6,7,8 digits, which is a
#simple transfer learning of MNIST data.
#Reference: https://github.com/geekyspartan/mnist_cnn_transfer_learning#mnist-cnn-using-transfer-learning

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import tensorflow as tf

#read MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:\hhh")

# To reset graph of tensorflow
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def getActivations(layer, stimuli):
    units = sess.run(layer, feed_dict={X: np.reshape(stimuli, [1, 784], order='F')})
    plotNNFilter(units)

#repreare trainning data and testing data
def prepare_data():
    global  dataset_1459_train, dataset_023678_train, X_1459_train, y_1459_train, X_1459_test, \
            y_1459_test, X_023678_test, y_023678_test,  X_023678_train, y_023678_train

    mask_1459_train = np.logical_or.reduce([mnist.train.labels == v for v in [1,4,5,9]])
    mask_1459_test  = np.logical_or.reduce([mnist.test.labels  == v for v in [1,4,5,9]])
    X_1459_train = np.compress(np.array(mask_1459_train), mnist.train.images, axis=0)
    _, y_1459_train = np.unique(mnist.train.labels[mask_1459_train], return_inverse=True) # 1,4,5,9 to 0,1,2,3
    dataset_1459_train = tf.data.Dataset.from_tensor_slices((X_1459_train,y_1459_train))

    X_1459_test = np.compress(np.array(mask_1459_test), mnist.test.images, axis=0)
    _, y_1459_test = np.unique(mnist.test.labels[mask_1459_test], return_inverse=True)

    mask_023678_train = np.logical_or.reduce([mnist.train.labels == v for v in [0,2,3,6,7,8]])
    mask_023678_test  = np.logical_or.reduce([mnist.test.labels  == v for v in [0,2,3,6,7,8]])
    X_023678_train = np.compress(np.array(mask_023678_train), mnist.train.images, axis=0)
    _, y_023678_train = np.unique(mnist.train.labels[mask_023678_train], return_inverse=True) # 1,4,5,9 to 0,1,2,3
    dataset_023678_train = tf.data.Dataset.from_tensor_slices((X_023678_train,y_023678_train))

    X_023678_test = np.compress(np.array(mask_023678_test), mnist.test.images, axis=0)
    _, y_023678_test = np.unique(mnist.test.labels[mask_023678_test], return_inverse=True)

#basic paprameters
height = 28
width = 28
channels = 1
n_inputs = height * width
n_outputs = 4

# 5 epochs, 10 batch size
n_epochs = 5
batch_size = 10
output_file1349 = "E:\hhh\output.txt"

reset_graph()
prepare_data()

# inputs layer
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

#step1: "conv1": conv2d 3x3x4, stride=1, ReLU, padding = "SAME"
#step2: conv2": conv2d 3x3x8, stride=2, ReLU, padding = "SAME"
with tf.name_scope("conv"):
    conv1 = tf.layers.conv2d(X_reshaped, 8, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding="SAME", name="conv1")
    conv2 = tf.layers.conv2d(conv1, 8, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, padding="SAME", name="conv2")

#step3: "pool": pool 2x2
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1,1,1,1], padding='SAME', name="pool")
    pool3_flat = tf.reshape(pool3, [-1, 14*14*8])

#step4: "fc1": fc 16
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, 16, name="fc1", activation=tf.nn.relu)

#step5: "fc2": fc 10
with tf.name_scope("fc2"):
    fc2 = tf.layers.dense(fc1, 10, name="fc2", activation=tf.nn.relu)

#step6: "softmax": xentropy loss, fc-logits = 4 (we have 4 classes...)
with tf.name_scope("output"):
    logits = tf.layers.dense(fc2, 4)
    Y_proba = tf.nn.softmax(logits, name="softmax")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()



print("Training...")
with tf.Session() as sess:
    init.run()
    batched_dataset = dataset_1459_train.batch(batch_size)

    iterator = batched_dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    for epoch in range(n_epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                X_batch, y_batch = sess.run(next_batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            except tf.errors.OutOfRangeError:
                # finished running through dataset
                break

        acc_train = accuracy.eval(feed_dict={X: X_1459_train, y: y_1459_train})
        acc_test = accuracy.eval(feed_dict={X: X_1459_test, y: y_1459_test})
        loss_1459_test = sess.run(loss, feed_dict={X: X_1459_test, y: y_1459_test})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test, "loss_1459_test:", loss_1459_test)

    print("Finished training")

    print("Saving...")
    save_path = saver.save(sess, "E:\hhh\my_model_1459.ckpt")

    # for example
    inferred = sess.run(Y_proba, feed_dict={X: [X_1459_test[0]]})



#reuse the weight of 1459 network
reset_graph()
prepare_data()

# restore the graph of 1459
restore_saver = tf.train.import_meta_graph("E:\hhh\my_model_1459.ckpt.meta")

# reuse the inputs (X,y)
X = tf.get_default_graph().get_tensor_by_name("inputs/X:0")
y = tf.get_default_graph().get_tensor_by_name("inputs/y:0")
# reuse the FC1 layer
fc1 = tf.get_default_graph().get_tensor_by_name("fc1/fc1/Relu:0")
# reuse the FC2 layer
fc2_reuse = tf.get_default_graph().get_tensor_by_name("fc2/fc2/Relu:0")

# continue the 023678 graph from FC2...
with tf.name_scope("fc2_023678"):
    fc2 = tf.layers.dense(fc2_reuse, 7, name="fc2_023678", activation=tf.nn.relu)

with tf.name_scope("softmax_023678"):
    logits = tf.layers.dense(fc2, 6, name="output_023678")
    Y_proba = tf.nn.softmax(logits, name="Y_proba_023678")

with tf.name_scope("train_023678"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(name="adam_023678")
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval_023678"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_023678"):
    init = tf.global_variables_initializer()
    new_saver = tf.train.Saver()

n_epochs = 10
batch_size = 10

print("Training 023678...")
with tf.Session() as sess:
    init.run()
    batched_dataset = dataset_023678_train.batch(batch_size)

    iterator = batched_dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    saver.restore(sess, "E:\hhh\my_model_1459.ckpt")

    for epoch in range(n_epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                X_batch, y_batch = sess.run(next_batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            except tf.errors.OutOfRangeError:
                # finished running through dataset
                break

        acc_train = accuracy.eval(feed_dict={X: X_023678_train, y: y_023678_train})
        acc_test = accuracy.eval(feed_dict={X: X_023678_test, y: y_023678_test})
        loss_023678_test = sess.run(loss, feed_dict={X: X_023678_test, y: y_023678_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test, "loss_023678_test:", loss_023678_test)

    print("Finished training")

    print("Saving...")
    save_path = saver.save(sess, "E:\hhh\my_model_023678.ckpt")

    # for example
    inferred = sess.run(Y_proba, feed_dict={X: [X_023678_test[1]]})
    plot_image(X_023678_test[1].reshape(28, 28), "Predicted %d, truth %d" % (np.argmax(inferred), y_023678_test[1]))
#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
training_epochs = 20
img_x = 28
img_y = 28
batch_size = 256
display_step = 1
step = 1
n_input = img_x*img_y
tunnel = 1
learning_rate = 0.0001
X=tf.placeholder(tf.float32,[None,img_y,img_x])
Y=tf.placeholder(tf.float32,[None, 10])
weights={
    'convin': tf.Variable(tf.truncated_normal([5,5,1,32],name='in')),
    'in': tf.Variable(tf.truncated_normal([196,128],name='in')),
    'out': tf.Variable(tf.truncated_normal([128,10],name='out'))
}

bias={
    'convin':tf.Variable(tf.truncated_normal([32],name='b1')),
    'in': tf.Variable(tf.truncated_normal([128],name='b1')),
    'out': tf.Variable(tf.truncated_normal([10],name='b2'))
}

def RNN(X,weights, bias):
    X = tf.reshape(X, (-1, 28,28,1))
    enconv_1 = tf.add(tf.nn.relu(tf.nn.conv2d(X, weights['convin'], strides=[1, 1, 1, 1], padding='SAME')),
                      bias['convin'], name='enconv_1')
    enpool_1 = tf.nn.max_pool(enconv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='enpool_1')
    X=tf.reshape(enpool_1,(-1,196))
    X=tf.matmul(X, weights['in'])+bias['in']
    X=tf.reshape(X, shape=(-1,32,128))
    cell=tf.contrib.rnn.BasicLSTMCell(128, forget_bias=1.0)
    init_state=cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    outputs,final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state)
    outputs=tf.unstack(outputs, axis=1)
    results = tf.matmul(outputs[-1], weights['out']) + bias['out']
    return results

pre=RNN(X,weights,bias)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pre))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pre, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, img_y, img_x])
            _, c=sess.run([optimizer,accuracy],feed_dict={
                X:batch_xs,Y:batch_ys
            })
            print("epoch", "%04d" % (epoch+1), "accuracy=","{:.4f}".format(c))
    print("optimization finished")



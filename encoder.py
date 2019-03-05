#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from numba import jit

mnist = input_data.read_data_sets("MNIST_data/",one_hot=False)

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
n_input = 784
X = tf.placeholder("float",[None, n_input])
examples_to_show=10

n_h1 = 128
n_h2 = 64
n_h3 = 10
n_h4 = 2

weights = {
    'e_h1': tf.Variable(tf.truncated_normal([n_input, n_h1])),
    'e_h2': tf.Variable(tf.truncated_normal([n_h1, n_h2])),
    'e_h3': tf.Variable(tf.truncated_normal([n_h2, n_h3])),
    'e_h4': tf.Variable(tf.truncated_normal([n_h3, n_h4])),
    'd_h1': tf.Variable(tf.truncated_normal([n_h4, n_h3])),
    'd_h2': tf.Variable(tf.truncated_normal([n_h3, n_h2])),
    'd_h3': tf.Variable(tf.truncated_normal([n_h2, n_h1])),
    'd_h4': tf.Variable(tf.truncated_normal([n_h1, n_input]))
}
biases = {
    'e_b1':tf.Variable(tf.random_normal([n_h1])),
    'e_b2':tf.Variable(tf.random_normal([n_h2])),
    'e_b3':tf.Variable(tf.random_normal([n_h3])),
    'e_b4':tf.Variable(tf.random_normal([n_h4])),
    'd_b1':tf.Variable(tf.random_normal([n_h3])),
    'd_b2':tf.Variable(tf.random_normal([n_h2])),
    'd_b3':tf.Variable(tf.random_normal([n_h1])),
    'd_b4':tf.Variable(tf.random_normal([n_input]))
}


#@jit
def encode(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['e_h1']),
                            biases['e_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['e_h2']),
                            biases['e_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['e_h3']),
                            biases['e_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['e_h4']),
                            biases['e_b4'])
    return layer_4

#@jit
def decode(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['d_h1']),
                            biases['d_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['d_h2']),
                            biases['d_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['d_h3']),
                            biases['d_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['d_h4']),
                            biases['d_b4']))
    return layer_4

e_output = encode(X)
d_output= decode(e_output)

y_pre = d_output
y_true = X

cost = tf.reduce_mean(tf.pow(y_true-y_pre,2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in  range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _, c= sess.run([optimizer,cost],feed_dict={X: batch_xs})
            print("epoch", "%04d" % epoch, i,  "cost=", "{:.9f}".format(c))
        if epoch % display_step == 0:
            print("epoch", "%04d"% (epoch+1), "cost=","{:.9f}".format(c))
    print("optimization finished")

    e_result = sess.run(
        e_output, feed_dict={X:mnist.test.images}
    )
    plt.scatter(e_result[:,0], e_result[:,1], c=mnist.test.labels)
    plt.colorbar()
    plt.show()

    encode_decode = sess.run(
        y_pre, feed_dict={X: mnist.test.images[:examples_to_show]})
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

training_epochs = 20
img_x = 28
img_y = 28
batch_size = 256
display_step = 1
step = 1
n_input = img_x*img_y
tunnel = 1
learning_rate = 0.0001
X = tf.placeholder('float',[None, n_input],name='X')
Y = tf.placeholder('float',[None, n_input],name='Y')
target = tf.placeholder(tf.float32,[None, 28,28,1])
mnist = input_data.read_data_sets("MNIST_data/",one_hot=False)
examples_to_show = 10


weight = {
    'w_c1': tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1),name='w_c1'),
    'w_c2': tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name='w_c2'), #   'w_a1': tf.Variable(tf.truncated_normal([7*7*64,1024])),    'w_da1': tf.Variable(tf.truncated_normal([1024,7*7*64])),
    'w_c3': tf.Variable(tf.truncated_normal([7*7, 4*4], stddev=0.1),name='w_c3'),
    'w_dc3': tf.Variable(tf.truncated_normal([4*4, 7*7], stddev=0.1),name='w_dc3'),
    'w_dc2': tf.Variable(tf.truncated_normal([5,5,64,32], stddev=0.1),name='w_dc2'),
    'w_dc1': tf.Variable(tf.truncated_normal([5,5,32,1], stddev=0.1),name='w_dc1'),
}

bias = {
    'b_c1': tf.Variable(tf.truncated_normal([32], stddev=0.1), name='b_c1'),
    'b_c2': tf.Variable(tf.truncated_normal([64], stddev=0.1),name='b_c2'),
    'b_c3': tf.Variable(tf.truncated_normal([4*4], stddev=0.1),name='b_c3'),
    'b_dc3': tf.Variable(tf.truncated_normal([7*7], stddev=0.1),name='b_dc3'),
    'b_dc2': tf.Variable(tf.truncated_normal([32], stddev=0.1),name='b_dc2'),
    'b_dc1': tf.Variable(tf.truncated_normal([1], stddev=0.1),name='b_dc1'),
}

x_image = tf.reshape(X,[-1, 28, 28, 1], name='x_image')
y_image = tf.reshape(Y,[-1, 28, 28, 1], name='x_image')
#x_image1 = x_image + 0.5*tf.random_normal(*x_image.shape)
#x_image1 = np.clip(x_image1, 0.0, 1.0)
enconv_1 = tf.add(tf.nn.relu(tf.nn.conv2d(x_image, weight['w_c1'], strides=[1, 1, 1, 1], padding='SAME')),bias['b_c1'],name='enconv_1')
enpool_1 = tf.nn.max_pool(enconv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='enpool_1')
enconv_2 = tf.add(tf.nn.relu(tf.nn.conv2d(enpool_1, weight['w_c2'], strides=[1, 1, 1, 1], padding='SAME')),bias['b_c2'], name='enconv_2')
enpool_2 = tf.nn.max_pool(enconv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
enpool_2 = tf.reshape(enpool_2, [-1, 7*7], name='enpool_2')
enlayer_1 = tf.add(tf.matmul(enpool_2, weight['w_c3']),
                               bias['b_c3'], name='enlayer_1')

delayer_1 = tf.nn.sigmoid(tf.add(tf.matmul(enlayer_1, weight['w_dc3']),
                               bias['b_dc3']),name='delayer_1')
delayer_1 = tf.reshape(delayer_1,[-1, 7, 7, 64],name='delayer_11')
depool_2 = tf.image.resize_nearest_neighbor(delayer_1,(14,14),name='depool_2')
deconv_2 = tf.nn.relu(tf.add(tf.nn.conv2d(depool_2, weight['w_dc2'], strides=[1, 1, 1, 1], padding='SAME'),bias['b_dc2']), name='deconv_2')
depool_1 = tf.image.resize_nearest_neighbor(deconv_2,(28,28),name='depool_1')
d_output = tf.add(tf.nn.conv2d(depool_1, weight['w_dc1'], strides=[1, 1, 1, 1], padding='SAME'),bias['b_dc1'], name='d_output')


y_pre = d_output
y_true = y_image

cost = tf.reduce_mean(tf.pow(y_true-y_pre,2),name='cost')
#print cost.shape
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
saver=tf.train.Saver()
init = tf.initialize_all_variables()

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        saver.restore(sess, 'save/convmodel.ckpt')
#        sess.run(init)
        total_batch = int(mnist.train.num_examples/batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs1 = batch_xs/2 + np.random.rand(batch_size, 784)
                _, c = sess.run([optimizer,cost],feed_dict={X: batch_xs1, Y:batch_xs})

                print("epoch", "%04d" % epoch, "i", "%04d" % (i + 1), "cost=", "{:.9f}".format(c))
            if epoch % display_step == 0:
                print("epoch", "%04d" % (epoch+1), "cost=","{:.9f}".format(c))
        print("optimization finished")
        saver_path=saver.save(sess, "save/convmodel.ckpt")
        print("Model saved in file:", saver_path)

    #    testimg = mnist.test.images[:examples_to_show]
    #    testimg = testimg+tf.random_normal(testimg.shape)
    #    testimg1=testimg.reshape([-1, 28, 28, 1])
        testimg = mnist.test.images[:examples_to_show]/2 + np.random.rand(examples_to_show, 784)
        encode_decode = sess.run(
            y_pre, feed_dict={X:testimg, Y: mnist.test.images[:examples_to_show]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(testimg[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        plt.show()
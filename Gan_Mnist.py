#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

training_epochs = 2
img_x = 28
img_y = 28
batch_size = 256
display_step = 1
step = 1
n_input = img_x*img_y
tunnel = 1
learning_rate = 0.00001
X = tf.placeholder('float',[None, 16],name='X')
x_data = tf.placeholder('float',[None, n_input],name='x_data')
target = tf.placeholder(tf.float32,[None, 28,28,1])
mnist = input_data.read_data_sets("MNIST_data/",one_hot=False)
examples_to_show = 10


weight_generator = {
    # 'w_c1': tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1),name='w_c1'),
    # 'w_c2': tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name='w_c2'), #   'w_a1': tf.Variable(tf.truncated_normal([7*7*64,1024])),    'w_da1': tf.Variable(tf.truncated_normal([1024,7*7*64])),
    # 'w_c3': tf.Variable(tf.truncated_normal([7*7, 4*4], stddev=0.1),name='w_c3'),
    'w_dc3': tf.Variable(tf.truncated_normal([4*4, 7*7], stddev=0.1),name='w_dc3'),
    'w_dc2': tf.Variable(tf.truncated_normal([3,3,1,16], stddev=0.1),name='w_dc2'),
    'w_dc1': tf.Variable(tf.truncated_normal([4,4,16,1], stddev=0.1),name='w_dc1'),
}

bias_generator = {
    # 'b_c1': tf.Variable(tf.truncated_normal([32], stddev=0.1), name='b_c1'),
    # 'b_c2': tf.Variable(tf.truncated_normal([64], stddev=0.1),name='b_c2'),
    # 'b_c3': tf.Variable(tf.truncated_normal([4*4], stddev=0.1),name='b_c3'),
    'b_dc3': tf.Variable(tf.truncated_normal([7*7], stddev=0.1),name='b_dc3'),
    'b_dc2': tf.Variable(tf.truncated_normal([16], stddev=0.1),name='b_dc2'),
    'b_dc1': tf.Variable(tf.truncated_normal([1], stddev=0.1),name='b_dc1'),
}

weight_discriminator = {
    'w_c1': tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1),name='w_c1'),
    'w_c2': tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name='w_c2'), #   'w_a1': tf.Variable(tf.truncated_normal([7*7*64,1024])),    'w_da1': tf.Variable(tf.truncated_normal([1024,7*7*64])),
    'w_c3': tf.Variable(tf.truncated_normal([7*7, 4*4], stddev=0.1),name='w_c3'),
    'w_dc3': tf.Variable(tf.truncated_normal([4*4, 7*7], stddev=0.1),name='w_dc3'),
    'w_dc2': tf.Variable(tf.truncated_normal([5,5,64,32], stddev=0.1),name='w_dc2'),
    'w_dc1': tf.Variable(tf.truncated_normal([5,5,32,1], stddev=0.1),name='w_dc1'),
    'w_out': tf.Variable(tf.truncated_normal([784,1],name='out'))
}

bias_discriminator = {
    'b_c1': tf.Variable(tf.truncated_normal([32], stddev=0.1), name='b_c1'),
    'b_c2': tf.Variable(tf.truncated_normal([64], stddev=0.1),name='b_c2'),
    'b_c3': tf.Variable(tf.truncated_normal([4*4], stddev=0.1),name='b_c3'),
    'b_dc3': tf.Variable(tf.truncated_normal([7*7], stddev=0.1),name='b_dc3'),
    'b_dc2': tf.Variable(tf.truncated_normal([32], stddev=0.1),name='b_dc2'),
    'b_dc1': tf.Variable(tf.truncated_normal([1], stddev=0.1),name='b_dc1'),
    'b_out': tf.Variable(tf.truncated_normal([1], stddev=0.1),name='b_dc1')
}


x_image = tf.reshape(X,[-1, 16], name='x_image')
#x_image1 = x_image + 0.5*tf.random_normal(*x_image.shape)
#x_image1 = np.clip(x_image1, 0.0, 1.0)
# enconv_1 = tf.add(tf.nn.relu(tf.nn.conv2d(x_image, weight_generator['w_c1'], strides=[1, 1, 1, 1], padding='SAME')),bias_generator['b_c1'],name='enconv_1')
# enpool_1 = tf.nn.max_pool(enconv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='enpool_1')
# enconv_2 = tf.add(tf.nn.relu(tf.nn.conv2d(enpool_1, weight_generator['w_c2'], strides=[1, 1, 1, 1], padding='SAME')),bias_generator['b_c2'], name='enconv_2')
# enpool_2 = tf.nn.max_pool(enconv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#enpool_2 = tf.reshape(enpool_2, [-1, 7*7], name='enpool_2')
# enlayer_1 = tf.add(tf.matmul(enpool_2, weight_generator['w_c3']),
#                                bias_generator['b_c3'], name='enlayer_1')
#
delayer_1 = tf.nn.relu(tf.add(tf.matmul(x_image, weight_generator['w_dc3']),
                                bias_generator['b_dc3']),name='delayer_1')
delayer_1 = tf.reshape(delayer_1,[-1, 7, 7, 1],name='delayer_1')
depool_2 = tf.image.resize_nearest_neighbor(delayer_1,(14,14),name='depool_2')
deconv_2 = tf.nn.relu(tf.add(tf.nn.conv2d(depool_2, weight_generator['w_dc2'], strides=[1, 1, 1, 1], padding='SAME'),bias_generator['b_dc2']), name='deconv_2')
depool_1 = tf.image.resize_nearest_neighbor(deconv_2,(28,28),name='depool_1')
x_generated = tf.nn.tanh(tf.add(tf.nn.conv2d(depool_1, weight_generator['w_dc1'], strides=[1, 1, 1, 1], padding='SAME'),bias_generator['b_dc1'], name='d_output'))
x_generated = tf.reshape(x_generated, (-1, 784))
print x_generated.shape

generator_param = [
    # weight_generator['w_c1'], bias_generator['b_c1'],
    #                weight_generator['w_c2'], bias_generator['b_c2'],
    #                weight_generator['w_c3'], bias_generator['b_c3'],
                   weight_generator['w_dc3'], bias_generator['b_dc3'],
                   weight_generator['w_dc2'], bias_generator['b_dc2'],
                   weight_generator['w_dc1'],bias_generator['b_dc1']]


x_in = tf.concat([x_data, x_generated], 0)
print x_in.shape
x_in = tf.reshape(x_in, (-1, 28, 28, 1))
enconv_1_discriminator = tf.add(tf.nn.relu(tf.nn.conv2d(x_in, weight_discriminator['w_c1'], strides=[1, 1, 1, 1], padding='SAME')),bias_discriminator['b_c1'],name='enconv_1')
enpool_1_discriminator = tf.nn.max_pool(enconv_1_discriminator, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='enpool_1')
enconv_2_discriminator = tf.add(tf.nn.relu(tf.nn.conv2d(enpool_1_discriminator, weight_discriminator['w_c2'], strides=[1, 1, 1, 1], padding='SAME')),bias_discriminator['b_c2'], name='enconv_2')
enpool_2_discriminator = tf.nn.max_pool(enconv_2_discriminator, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
enpool_2_discriminator = tf.reshape(enpool_2_discriminator, [-1, 7*7], name='enpool_2')
enlayer_1_discriminator = tf.add(tf.matmul(enpool_2_discriminator, weight_discriminator['w_c3']),
                               bias_discriminator['b_c3'], name='enlayer_1')

delayer_1_discriminator = tf.nn.relu(tf.add(tf.matmul(enlayer_1_discriminator, weight_discriminator['w_dc3']),
                               bias_discriminator['b_dc3']),name='delayer_1')
delayer_1_discriminator = tf.reshape(delayer_1_discriminator,[-1, 7, 7, 64],name='delayer_11')
depool_2_discriminator = tf.image.resize_nearest_neighbor(delayer_1_discriminator,(14,14),name='depool_2')
deconv_2_discriminator = tf.nn.relu(tf.add(tf.nn.conv2d(depool_2_discriminator, weight_discriminator['w_dc2'], strides=[1, 1, 1, 1], padding='SAME'),bias_discriminator['b_dc2']), name='deconv_2')
depool_1_discriminator = tf.image.resize_nearest_neighbor(deconv_2_discriminator,(28,28),name='depool_1')
d_output_discriminator = tf.add(tf.nn.conv2d(depool_1_discriminator, weight_discriminator['w_dc1'], strides=[1, 1, 1, 1], padding='SAME'),bias_discriminator['b_dc1'], name='d_output')
d_output_discriminator = tf.reshape(d_output_discriminator, (-1, 784))

d_output = tf.add(tf.matmul(d_output_discriminator,weight_discriminator['w_out']), bias_discriminator['b_out'])
y_data = tf.nn.sigmoid(tf.slice(d_output, [0, 0], [batch_size, -1], name=None))
y_generated = tf.nn.sigmoid(tf.slice(d_output, [batch_size, 0], [-1, -1], name=None))

discriminator_param = [weight_discriminator['w_c1'], bias_discriminator['b_c1'],
                   weight_discriminator['w_c2'], bias_discriminator['b_c2'],
                   weight_discriminator['w_c3'], bias_discriminator['b_c3'],
                   weight_discriminator['w_dc3'], bias_discriminator['b_dc3'],
                   weight_discriminator['w_dc2'], bias_discriminator['b_dc2'],
                   weight_discriminator['w_dc1'],bias_discriminator['b_dc1'],
                   weight_discriminator['w_out'], bias_discriminator['b_out']]


d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
d_loss1= tf.reduce_mean(d_loss)
g_loss = - tf.log(y_generated)
g_loss1= tf.reduce_mean(g_loss)
d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=discriminator_param)
g_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=generator_param)
init = tf.initialize_all_variables()
saver=tf.train.Saver()

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        saver.restore(sess, 'save2/Ganmodel.ckpt')
#        sess.run(init)
        total_batch = int(mnist.train.num_examples/batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = 2 * batch_xs.astype(np.float32) - 1
                z_value = np.random.normal(0, 1, size=(256, 16)).astype(np.float32)
                _, d = sess.run([d_optimizer, d_loss1],feed_dict={X: z_value, x_data:batch_xs})
                for j in range(5):
                    z_value = np.random.normal(0, 1, size=(256, 16)).astype(np.float32)
                    _, g = sess.run([g_optimizer, g_loss1], feed_dict={X: z_value, x_data: batch_xs})

                #_, g = sess.run([g_optimizer, g_loss1], feed_dict={X: z_value, x_data: batch_xs})

                print("epoch", "%04d" % epoch, "i", "%04d" % (i + 1), "d_cost=", "{:.9f}".format(d), "g_cost=", "{:.9f}".format(g))


        print("optimization finished")
        saver_path=saver.save(sess, 'save2/Ganmodel.ckpt')
        print("Model saved in file:", saver_path)

    #    testimg = mnist.test.images[:examples_to_show]
    #    testimg = testimg+tf.random_normal(testimg.shape)
    #    testimg1=testimg.reshape([-1, 28, 28, 1])
        testimg = np.random.normal(0, 1, size=(examples_to_show, 16)).astype(np.float32)
        testimg1 = mnist.test.images[:examples_to_show]
        generation = sess.run(
            x_generated, feed_dict={X:testimg, x_data: mnist.test.images[:examples_to_show]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        generation = (generation+1) / 2
        print generation.shape
        generation= generation[0:10]
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(testimg1[i], (28, 28)))
            a[1][i].imshow(np.reshape(generation[i], (28, 28)))
        plt.show()
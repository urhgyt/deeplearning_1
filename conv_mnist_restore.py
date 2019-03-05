#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


training_epochs = 0
img_x = 28
img_y = 28
batch_size = 256
display_step = 1
step = 1
n_input = img_x*img_y
tunnel = 1
learning_rate = 0.001


mnist = input_data.read_data_sets("MNIST_data/",one_hot=False)
examples_to_show = 15
saver= tf.train.import_meta_graph('save/convmodel.ckpt.meta')
graph=tf.get_default_graph()
X=graph.get_tensor_by_name('X:0')
Y=graph.get_tensor_by_name('Y:0')
d_output=tf.get_default_graph().get_tensor_by_name('d_output:0')
cost=tf.get_default_graph().get_tensor_by_name('cost:0')
optimizer = tf.get_default_graph().get_operation_by_name('Adam')

config=tf.ConfigProto(allow_soft_placement=True)
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
        saver.restore(sess, 'save/convmodel.ckpt')
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

        saver.restore(sess, 'save/convmodel.ckpt')

        testimg=mnist.test.images[:examples_to_show]/2+ np.random.rand(examples_to_show, 784)
        encode_decode = sess.run(
            d_output,
            feed_dict={X: testimg, Y:mnist.test.images[:examples_to_show]})

        f, a = plt.subplots(2, 15, figsize=(15, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(testimg[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        plt.show()

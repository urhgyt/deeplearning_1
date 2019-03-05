#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import test2


img_num=250
training_epochs = 0
img_x = 256
img_y = 1024
batch_size = 16
display_step = 1
step = 1
n_input = img_x*img_y
tunnel = 1
learning_rate = 0.001
X=tf.placeholder(tf.float32,[None,img_y,img_x])
Y=tf.placeholder(tf.float32,[None,img_y,img_x])
#batch_size=tf.placeholder(dtype=tf.int32)
examples_to_show=8
total_batch = int(img_num / batch_size)
data_path='iris_contact.tfrecords'


weights={
    'conv_in1': tf.Variable(tf.truncated_normal([3,3,1,32],name='conv_in1')),
    'conv_in2': tf.Variable(tf.truncated_normal([5,5,32,64],name='conv_in2')),
    'conv_in3': tf.Variable(tf.truncated_normal([5,5,64,64],name='conv_in2')),

#    'gener': tf.Variable(tf.truncated_normal([3, 128*32, 128], name='gener')),
#     'rnn_in': tf.Variable(tf.truncated_normal([196*32,196],name='rnn_in')),
#    'rnn_out': tf.Variable(tf.truncated_normal([16*16*64, 16*16*64],name='rnn_out')),
    'conv_out1': tf.Variable(tf.truncated_normal([5,5,64,64],name='conv_out1')),
    'conv_out2': tf.Variable(tf.truncated_normal([5, 5, 64, 32], name='conv_out2')),
    'conv_out3': tf.Variable(tf.truncated_normal([3, 3, 32, 1], name='conv_out3'))
}

bias={
    'b1':tf.Variable(tf.truncated_normal([32],name='b_1')),
    'b2':tf.Variable(tf.truncated_normal([64],name='b_2')),
    'b3':tf.Variable(tf.truncated_normal([64],name='b_3')),
    'b4': tf.Variable(tf.truncated_normal([64],name='b_4')),
    'b5': tf.Variable(tf.truncated_normal([32],name='b_5')),
    'b6':tf.Variable(tf.truncated_normal([1],name='b_6')),
    'b_out': tf.Variable(tf.truncated_normal([1],name='b_out')),
}


def decoder(results, weights, bias):
    results = tf.reshape(results, (-1,16,16,64))
    depool_1 = tf.image.resize_nearest_neighbor(results, (32, 32), name='depool_1')
    deconv_1 = tf.add(tf.nn.conv2d(depool_1, weights['conv_out1'], strides=[1, 1, 1, 1], padding='SAME'), bias['b4'],
                      name='deconv_1')

    depool_2 = tf.image.resize_nearest_neighbor(deconv_1, (64, 64), name='depool_2')
    deconv_2 = tf.add(tf.nn.conv2d(depool_2, weights['conv_out2'], strides=[1, 1, 1, 1], padding='SAME'), bias['b5'],
                      name='deconv_2')

    depool_3 = tf.image.resize_nearest_neighbor(deconv_2, (256, 256), name='depool_3')
    deconv_3 = tf.add(tf.nn.conv2d(depool_3, weights['conv_out3'], strides=[1, 1, 1, 1], padding='SAME'), bias['b6'],
                      name='deconv_3')
    d_output=tf.reshape(deconv_3, (-1, 256, 256),name='d_output')

    return d_output


def RNN(X,weights, bias,batch_size):
    X = tf.reshape(X, (-1, 256,256,1))
    enconv_1 = tf.add(tf.nn.relu(tf.nn.conv2d(X, weights['conv_in1'], strides=[1, 1, 1, 1], padding='SAME')),
                      bias['b1'], name='enconv_1')
    enpool_1 = tf.nn.avg_pool(enconv_1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='enpool_1')

    enconv_2 = tf.add(tf.nn.relu(tf.nn.conv2d(enpool_1, weights['conv_in2'], strides=[1, 1, 1, 1], padding='SAME')),
                      bias['b2'], name='enconv_2')
    enpool_2 = tf.nn.avg_pool(enconv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='enpool_2')
    enconv_3 = tf.add(tf.nn.relu(tf.nn.conv2d(enpool_2, weights['conv_in3'], strides=[1, 1, 1, 1], padding='SAME')),
                      bias['b3'], name='enconv_3')
    enpool_3 = tf.nn.avg_pool(enconv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='enpool_3')

    X=tf.reshape(enpool_3,(-1,16*16*64))
    X=tf.reshape(X,(-1,4,16*16*64))
#    inputs=tf.nn.embedding_lookup(X, [1,2])
#    new=tf.Variable(tf.truncated_normal([2,196], name='cells'))
#    inputs=tf.stack([inputs, new], axis=0)
#    inputs=tf.reshape(inputs, (-1, 4, 196))
    cell_fw=tf.contrib.rnn.BasicLSTMCell(16*16*64, forget_bias=1.0, state_is_tuple=True)
    cell_bw=tf.contrib.rnn.BasicLSTMCell(16*16*64, forget_bias=1.0, state_is_tuple=True)
    init_statefw=cell_fw.zero_state(batch_size=batch_size, dtype=tf.float32)
    init_statebw = cell_bw.zero_state(batch_size=batch_size, dtype=tf.float32)

    outputs,final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=X, initial_state_fw=init_statefw,initial_state_bw=init_statebw, time_major=False)
#    outputs = tf.concat(out, 2)
#    outputs=tf.unstack(out, axis=1)
    outputs = tf.stack(outputs, axis=0)
    outputs = tf.transpose(tf.add(outputs[0],outputs[1]),(1,0,2))

    print outputs
    # results1 = tf.matmul(outputs[0], weights['rnn_out']) + bias['b_out']
    # results2 = tf.matmul(outputs[1], weights['rnn_out']) + bias['b_out']
    # results3 = tf.matmul(outputs[2], weights['rnn_out']) + bias['b_out']
    # results4 = tf.matmul(outputs[3], weights['rnn_out']) + bias['b_out']
    results1=decoder(outputs[0],weights,bias)
    results2=decoder(outputs[1],weights,bias)
    results3=decoder(outputs[2],weights,bias)
    results4=decoder(outputs[3],weights,bias)
    results=[results1,results2,results3,results4]
    return results
#    results = tf.stack([results1,results2], name='stack1')


def get_datasets_batch(data_path, batch_size):

    feature = {
        'img_label': tf.FixedLenFeature([], tf.string),
        'img_raw': tf.FixedLenFeature([], tf.string)
    }
    filename_queue = tf.train.string_input_producer([data_path])

    # 定义一个 reader ，读取下一个 record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 解析读入的一个record
    features = tf.parse_single_example(serialized_example, features=feature)

    # 将字符串解析成图像对应的像素组
    img_raw = tf.decode_raw(features['img_raw'],out_type=tf.float32)

    # 将标签转化成int32
    img_label = tf.decode_raw(features['img_label'],out_type=tf.float32)

    # 这里将图片还原成原来的维度
    img_raw = tf.reshape(img_raw, [1024, 256])
    img_label = tf.reshape(img_label, [1024, 256])
#    print img_raw, img_label
    images, labels = tf.train.shuffle_batch([img_raw, img_label], batch_size=batch_size, capacity=20,
                                            min_after_dequeue=10)

    return images, labels


pre=RNN(X,weights,bias,batch_size)
Y1=tf.transpose(tf.reshape(Y,(-1,4, 256, 256)),perm=[1, 0, 2, 3])
print Y.shape
loss=tf.pow(Y1[0]-pre[0],2)+tf.pow(Y1[1]-pre[1],2)+tf.pow(Y1[2]-pre[2],2)+tf.pow(Y1[3]-pre[3],2)

cost=tf.reduce_mean(loss,name='cost')
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#correct_pred = tf.equal(tf.argmax(pre, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    with tf.device("/gpu:0"):
#        saver.restore(sess, 'save1/rnnmodel.ckpt')
        sess.run(init)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = get_datasets_batch(data_path,batch_size=batch_size)
                batch_xs = batch_xs.reshape([batch_size, img_y, img_x])
                _, c = sess.run([optimizer,cost],feed_dict={X: batch_xs, Y:batch_ys})

                print("epoch", "%04d" % epoch, "i", "%04d" % (i + 1), "cost=", "{:.9f}".format(c))
            if epoch % display_step == 0:
                print("epoch", "%04d" % (epoch+1), "cost=","{:.9f}".format(c))
            saver_path=saver.save(sess, "save1/rnnmodel.ckpt")
        print("optimization finished")

#        print("Model saved in file:", saver_path)
#        saver_path=saver.save(sess, "save1/rnnmodel.ckpt")
#        print("Model saved in file:", saver_path)

    #    testimg = mnist.test.images[:examples_to_show]
    #    testimg = testimg+tf.random_normal(testimg.shape)
    #    testimg1=testimg.reshape([-1, 28, 28, 1])
        testimg =get_datasets_batch(data_path,batch_size=examples_to_show)
        testimg = tf.reshape(testimg,(examples_to_show, 256, img_x))
        batch_size=8
        encode_decode = sess.run(
            pre, feed_dict={X:testimg, Y: testimg})
        f, a = plt.subplots(2, 8, figsize=(8, 2))
        img=np.reshape(encode_decode, (-1,2, 256, 256))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(testimg[i], ( 256, 256)))
            a[1][i].imshow(np.reshape(img[i%4,i/4], (256, 256)))
        plt.show()

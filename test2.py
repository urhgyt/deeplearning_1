#coding:utf-8
import tensorflow as tf
from PIL import Image
import numpy as np

data_path='iris_contact.tfrecords'

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

    with tf.Session() as sess:
        images, labels = tf.train.shuffle_batch([img_raw, img_label], batch_size=batch_size, capacity=20,
                                                min_after_dequeue=10)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # 启动多线程处理输入数据
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        img,lab = sess.run([images,labels])

     #   img1.show()
            # 关闭线程
        coord.request_stop()
        coord.join(threads)
        sess.close()
        return img, lab

img,lab=get_datasets_batch(data_path,batch_size=10)
print img.shape, img.dtype
print lab.shape, lab.dtype
img = lab.astype('uint8')
img = Image.fromarray(img[0])
#    img = Image.fromarray(img[0])
img.show()
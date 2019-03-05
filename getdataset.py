import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd='/home/urhgyt/videos/'

writer= tf.python_io.TFRecordWriter("iris_contact.tfrecords")
img_num=0
img_label=np.zeros((256,256))
img_raw=np.zeros((256,256))

for img_num in range(5000):
    img_num += 1
    img_path = cwd+"video" + "_%05d.jpg" % img_num
    img = Image.open(img_path)
    img=img.convert('L')
    print 'loading',img_num
    if img_num%2 ==1:
        img_label = np.concatenate((img_label, img), axis=0)
    else:
        img_raw = np.concatenate((img_raw, img), axis=0)
    if img_num%8==0 :
        img_raw=img_raw[256:].astype('float32')
        img_raw1=img_raw
        img_label =img_label[256:].astype('float32')
        img_label1 = img_label
        print img_label.dtype, img_raw.dtype
        img_raw = img_raw.tobytes()
        img_label = img_label.tobytes()
    #plt.imshow(img) # if you want to check you image,please delete '#'
    #plt.show()
        example = tf.train.Example(features=tf.train.Features(feature={
            "img_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
        img_label=np.zeros((256,256))
        img_raw=np.zeros((256,256))
        if img_num>=100:
            print img_raw1.shape
            img_raw1=img_raw1.astype('uint8')
            img_raw1=Image.fromarray(img_raw1)
            img_label1 = img_label1.astype('uint8')
            img_label1 = Image.fromarray(img_label1)
            img_raw1.show()
            img_label1.show()
            break

writer.close()
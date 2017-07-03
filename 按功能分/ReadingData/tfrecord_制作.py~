"""
图片地址：D:\Python\data\dog目录下有两个文件夹表示2类 chihuahua ，husky

1 先聊一下tfrecord, 这是一种将图像数据和标签放在一起的二进制文件，能更好的利用内存，在tensorflow中快速的复制，移动，读取，存储 等等..

这里注意，tfrecord会根据你选择输入文件的类，自动给每一类打上同样的标签
如在本例中，只有0,1 两类
"""
########################################################
#制作tfrecored文件
import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt 
import numpy as np
 
cwd='D:\Python\data\dog\\'
classes={'husky','chihuahua'} #人为 设定 2 类
writer= tf.python_io.TFRecordWriter("dog_train.tfrecords") #要生成的文件
 
for index,name in enumerate(classes):
    class_path=cwd+name+'\\'
    for img_name in os.listdir(class_path): #列出指定目录的文件
        img_path=class_path+img_name #每一个图片的地址
 
        img=Image.open(img_path)
        img= img.resize((128,128)) #统一为128x128x3
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串
 
writer.close()

#########################################################
#读取TFRECORD文件
def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)#解码string->uint8
    img = tf.reshape(img, [128, 128, 3])  #reshape为128*128的3通道图片或 img.set_shape([128,128,3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label

###########################################################################
#显示tfrecord格式的图片
filename_queue = tf.train.string_input_producer(["dog_train.tfrecords"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [128, 128, 3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(20):#共20张图片
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
'''
或
plt.imshow(img)
plt.show()
encode_img=tf.image.convert_image.encode_jpeg(image)
with tf.gfile.GFile(cwd+str(i)+'_''Label_'+str(l),"wb") as f:
    f.write(encode_img.eval())   
'''
    coord.request_stop()
    coord.join(threads)


























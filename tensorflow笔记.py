#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:31:10 2017

@author: keetsky
"""
#%%

#################################################
1.   Variable


# 定义变量  hidden1/weights  hidden1/biases
tf.name_scope('hidden1'):
    weights=tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],stddev=1.0/ math.sqrt(float(IMAGE_PIXELS))),name='weights')
    biases=tf.Variable(tf.zeros([hidden1_units]),name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

#变量初始化
init = tf.global_variables_initializer()
sess.run(init)

#########################################################3##
2.   通用
tf.linspace(-1., 1., 500)
zero_tsr = tf.zeros([row_dim, col_dim]) 
ones_tsr = tf.ones([row_dim, col_dim])
filled_tsr = tf.fill([row_dim, col_dim], 42)
constant_tsr = tf.constant([1,2,3])  , tf.constant(42, [row_dim, col_dim])
zeros_similar=tf.zeros_like(constant_tsr)
ones_sililar=tf.ones_like(constant_tsr)
linear_tsr=tf.linspace(start=0,stop=1,num=3) 与numpy中的linspace相似
Integer_seq_tsr=tf.range(start=6,limit=15,delta=3)    range()

#稀疏矩阵定义，分别在a[0][1]=6,a[2][4]=0.5
a = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])

#随机
Randunif_tsr=tf.random_uniform([row_dim,col_dim],minval=0,maxval=1)
Randnorm_tsr=tf.random_normal([row_dim,col_dim],mean=0.0,stddev=1.0)
Runcnorm_tsr=tf.truncated_normal([row_dim, col_dim],mean=0.0, stddev=1.0)
Shuffled_output=tf.random_shuffle(input_tensor)
Cropped_output=tf.random_crop(input_tensor,crop_size)
Cropped_image=tf.random_crop(my_image,[height/2,width/2,3])  随机剪切
#
Numpy->tensor: convert_to_tensor()
#矩阵
idnetity_matrix=tf.diag([1.0,1.0,1.0])
A=tf.truncated_normal([2,3])
B=tf.fill([2,3],5.0)
C=tf.random_uniform([3,2])
D=tf.convert_to_tensor(np.array([1.,2.,3.]))
tf.matmul(A,B)
tf.transpose(C)
tf.matrix_determinant(D)#行列式
tf.matrix_inverse(D)
tf.cholesky(D)
tf.self_adjoint_eig(D)#7.特征值和特征向量，第一行为特征值其他的行列为特征向量
#运算
abs,ceil,cos,exp,floor,inv,log,maximum,minimum,neg,pow,round,rsqrt,sign,sin,sqrt,square
,digamma,erf,erfc,igamma,igammac,lbeta,lgamma,squared_difference
#激活函数
tf.nn.relu()    即max(0,x)
tf.nn.relu6()   即:min(max(0,x),6)
tf.nn.sigmoid()  即：1/(1+exp(-x))      值域[0,1]
tf.nn.tanh()	   即：(exp(x)-exp(-x))/(exp(x)+exp(-x)）     值域 [-1,1]
tf.nn.softsign()   即：x/(abs(x)+1)
tf.nn.softplus()     即:log(exp(x)+1)
tf.nn.elu()		即：x<0 ? exp(x)+1:x  
#损失函数
1.L2 norm loss
l2_y_vals = tf.square(target - x_vals)
2.L1 norm loss
l1_y_vals = tf.abs(target - x_vals)
3.Pseudo-huber loss
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.mul(tf.square(delta1), tf.sqrt(1. +tf.square((target - x_vals)/delta1)) - 1.)
4.classfication loss

5.hinge loss
hinge_y_vals = tf.maximum(0., 1. - tf.mul(target, x_vals))l
6.cross-entropy
xentropy_y_vals = - tf.mul(target, tf.log(x_vals)) - tf.mul((1. -target), tf.log(1. - x_vals))
7.sigmoid cross entropy loss
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(x_vals, targets)
8.weighted cross entropy
eight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
9.softmax cross-entropy
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits, target_dist)
10.sparse softmax cross-entropy
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_
logits(unscaled_logits, sparse_target_dist)  
	
#############################################################
#将维度为1的去掉
tf.squeeze(input_array)
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
Or, to remove specific size 1 dimensions:

# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]，去掉第2和4维
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
################################################################
3. 将python数据转化为tu
feature_column_data = [1, 2.4, 0, 9.9, 3, 120]
feature_tensor = tf.constant(feature_column_data)
#or
a=np.zeros((3,3))
ta=tf.convert_to_tensor(a)

##################################################
2. Graph 设置默认图
with tf.Graph().as_default():
#重设
tf.reset_default_graph()
#
from tensorflow.python.framework import ops
ops.reset_default_graph()

##################################################
5. DATA 数据处理  csv bin tfrecords


#下载数据
if not os.path.exists("iris_training.csv"):
  raw = urllib.urlopen("http://download.tensorflow.org/data/iris_training.csv").read()
  with open(iris_training.csv,'w') as f:
    f.write(raw)
#加载csv数据iris
'''
30	4	setosa	versicolor	virginica
5.9	3	4.2	1.5	1
6.9	3.1	5.4	2.1	2
'''
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename="iris_training.csv",	#文件名
    target_dtype=np.int,		#numpy datatype 一般为分类的种类，整数
    features_dtype=np.float32)		#numpy datatype 特征数据的类型
x = tf.constant(training_set.data)      #用.data的方式访问其特征，或者使用.target的方式访问其标签
_y= tf.constant(training_set.target)
#加载csv数据boston_hosing_prediction
'''
100	9	CRIM	ZN	INDUS	NOX	RM	AGE	DIS	TAX	PTRATIO	MEDV
0.13587	0	10.59	0.489	6.064	59.1	4.2392	277	18.6	24.4		
0.08664	45	3.44	0.437	7.178	26.3	6.4798	398	15.2	36.4		
'''
tf.logging.set_verbosity(tf.logging.INFO)  #设置载入logging
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"
training_set = pd.read_csv("boston_train.csv",skipinitialspace=True,skiprows=1, names=COLUMNS)#读取数据，跳过第一行，按COLUMMS读取数据,字典格式
feature_cols = {k: tf.constant(train_set[k].values) for k in FEATURES}
labels = tf.constant(train_set[LABEL].values)





''' TFrecorder 保存和读取 ，三种数据特征格式：BytesList FloatList Int64List '''
#TFrecorder
#先遍历文件夹，然后写入到tfrecorder中假定文件夹目录为data/face_data/s1 ,s2 ... s10/1.bmp,2.bmp...10.bmp
import tensorflow as tf
import numpy as np 
import PIL import Image
import os
cwd=os.getcwd()#获取当前目录路径
root=cwd+"/data/face_data" #获取数据目录路径
TFwriter=tf.python_io_TFRecordWriter("./data/faceTF.tfrecords")#创建写TFrecord，设定写文件
for className in os.listdir(root): #[s1,s2,...,s10]
    label=int(className[1:])#s1->int(1) 类别y
    classPath=root+"/"+className+"/" #类别目录 ./data/face_data/s1
    for parent,dirnames,filenames in os.walk(classPath):#(["./data/face_data/s1"],[],["1.bmp","2.bmp",...,"10.bmp"])
        for filename in filenames:# 1.bmp   x1
	    imgPath=classPath+"/"+filename#图片路径
	    img=Image.open(imgPath)#打开指定路径图片
  	    imgRaw=img.tobytes()#转换为二进制格式
  	    example=tf.train.Example(features=tf.train.Features(feature={"label":tr.train.Feature(int64_list=tf.train.Int64List(value=[label])),"img":tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))}))
	    TFwriter.write(example.SerializeToSring())#将一个样例x1 y1 装换为Example protocol buffer，然后写入TFrecord 中 
TFwriter.close()
#读tfrecord
fileNameQue = tf.train.string_input_producer(["./data/faceTF.tfrecords"])#创建一个列端维护输入文件列表
reader = tf.TFRecordReader()#创建一个reader来读取TFrecord文件中的x y
key,value = reader.read(fileNameQue)#读取一个样例
features = tf.parse_single_example(value,features={ 'label': tf.FixedLenFeature([], tf.int64),
                                           'img' : tf.FixedLenFeature([], tf.string),})#解析一个样例Int64List->int64 BytesList->string
img = tf.decode_raw(features["img"], tf.uint8)#进一步解析 string->unit8
label = tf.cast(features["label"], tf.int32)#int64->int32

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()#启动多线程
    threads = tf.train.start_queue_runners(coord=coord)
    ImgArrs=[]
    Labels=[]
    for i in range(100):#读取100个样例
        imgArr,lab = sess.run([img,label])
        ImgArrs.append(imgArr)
	Labels.append(lab)

    coord.request_stop()
    coord.join(threads)

##################################################
10.  可视化  summary

'''$ tensorboard --logdir='logs/'

再在网页中输入链接：127.0.1.1:6006 即可获得展示：'''
#%%可视化    http://blog.csdn.net/l18930738887/article/details/55000008
'''显示神经网络的结构'''
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs
# Make up some real data
#x_data = np.linspace(-1,1,300)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise
# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1],name='input_x')
    ys = tf.placeholder(tf.float32, [None, 1],name='input_y')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)


# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# save the logs
writer = tf.summary.FileWriter("logs/", sess.graph)    #只加了这部分
sess.run(tf.global_variables_initializer())




#%%
'''loss及权重迭代可视化'''
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)     #变量可视化
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
# Make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1],name='input_x')
    ys = tf.placeholder(tf.float32, [None, 1],name='input_y')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)


# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)    #loss（常量）可视化
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()      #合并到summary中
# save the logs
writer = tf.summary.FileWriter("logs/", sess.graph)  #选定可优化存储目录
sess.run(tf.global_variables_initializer())
for step in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if step % 50 == 0:
        # to see the step improvement
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, step)    #写入log文件夹

##########################################################################
11. contrib.learn   DNN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets   如果打开失败可手动到网站下载csv
#特征x:Sepal Length 	Sepal Width 	Petal Length 	Petal Width 	类别y:Species
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w+") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w+") as f:
      f.write(raw)

 # Load datasets. Load dataset from CSV file with a header row.# 使用Tensorflow内置的方法进行数据加载
#，target_type是最终的label的类型，这里只有012三个取值，所以用int,用.data的方式访问其特征，或者使用.target的方式访问其标签
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,     # flower species
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)
#tf.contrib.learn提供了很多已经实现的模型，这里被称作Estimators，通过这些已经定义好的模型，
#你可以快速的基于这些模型和你的数据做一些分析工作。
#在这里我们将会创建一个DNN分类器，这里用到了DNNClassifier。
  # Specify that all features have real-value data
  ## 每行数据4个特征，都是real-value的
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  ## 构建一个DNN分类器，3层，其中每个隐含层的节点数量分别为10，20，10，目标的分类3个，并且指定了保存位置
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="tmp/iris_model")
  # Define the training inputs,
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # Fit model.# 指定数据，以及训练的步数
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]
  

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.# 直接创建数据来进行预测
  def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
 #从tmp/iris_model加载的模型中进行预测
  predictions = list(classifier.predict(input_fn=new_samples))

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))
main()
#######
"""DNNRegressor with custom input_fn for Housing dataset.
Logging and monitoring Basics with tf.contrilb.learn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels



  # Load datasets
training_set = pd.read_csv("/home/keetsky/Desktop/tensorflow_learn/boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("/home/keetsky/Desktop/tensorflow_learn/boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
prediction_set = pd.read_csv("/home/keetsky/Desktop/tensorflow_learn/boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)

  # Feature cols
feature_cols = [tf.contrib.layers.real_valued_column(k)
                for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir="/tmp/boston_model")

  # Fit
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

  # Score accuracy
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

  # Print out predictions
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
  # .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))


##################################################################




















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf8 -*-
"""
Created on Thu May  4 11:31:10 2017

@author: keetsky
"""

#%%basic
"""comp with numpy
a=tf.zeros(2,2)                     np.zeros
b=tf.ones(2,2)                      np.ones
a.get_shape()                       np.shape
tf.reshape(a,(1,4)).eval()          np.reshape
tf.matmul(a,b)                      np.dot(a,b)

a=tf.constant(5.0)                  a=5.0
w2=tf.Variable(3.0)
tf.add
tf.mul
w2=tf.assign(w2,4.0)
input=tf.placeholder(tf.float32)
tf.zeros_like
tf.ones_like
tf.fill
tf.range
tf.linspace


tf.random_normal
tf.truncated_normal
tf.random_uniform
tf.random_shuffle

1、tf.ones(shape,type=tf.float32,name=None)
     tf.ones([2, 3], int32) ==> [[1, 1, 1], [1, 1, 1]]

2、tf.zeros(shape,type=tf.float32,name=None)
     tf.zeros([2, 3], int32) ==> [[0, 0, 0], [0, 0, 0]]

3、tf.ones_like(tensor,dype=None,name=None)
     新建一个与给定的tensor类型大小一致的tensor，其所有元素为1。
     # 'tensor' is [[1, 2, 3], [4, 5, 6]] 
     tf.ones_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]

4、tf.zeros_like(tensor,dype=None,name=None)
     新建一个与给定的tensor类型大小一致的tensor，其所有元素为0。
     # 'tensor' is [[1, 2, 3], [4, 5, 6]] 
     tf.ones_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]

5、tf.fill(dim,value,name=None)
     创建一个形状大小为dim的tensor，其初始值为value
     # Output tensor has shape [2, 3]. 
     fill([2, 3], 9) ==> [[9, 9, 9] 
                                 [9, 9, 9]]

6、tf.constant(value,dtype=None,shape=None,name='Const')
     创建一个常量tensor，先给出value，可以设定其shape
     # Constant 1-D Tensor populated with value list. 
     tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7] 
    # Constant 2-D tensor populated with scalar value -1. 
     tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.] [-1. -1. -1.]

7、tf.linspace(start,stop,num,name=None)
     返回一个tensor，该tensor中的数值在start到stop区间之间取等差数列（包含start和stop），如果num>1则差值为(stop-start)/(num-1)，以保证最后一个元素的值为stop。
     其中，start和stop必须为tf.float32或tf.float64。num的类型为int。
     tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0 11.0 12.0]

8、tf.range(start,limit=None,delta=1,name='range')
     返回一个tensor等差数列，该tensor中的数值在start到limit之间，不包括limit，delta是等差数列的差值。
     start，limit和delta都是int32类型。
     # 'start' is 3 
     # 'limit' is 18 
     # 'delta' is 3
     tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15] 
     # 'limit' is 5 start is 0
     tf.range(start, limit) ==> [0, 1, 2, 3, 4]

9、tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
     返回一个tensor其中的元素的值服从正态分布。
     seed: A Python integer. Used to create a random seed for the distribution.See set_random_seed for behavior。

10、tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    返回一个tensor其中的元素服从截断正态分布（？概念不懂，留疑）

11、tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)
    返回一个形状为shape的tensor，其中的元素服从minval和maxval之间的均匀分布。

12、tf.random_shuffle(value,seed=None,name=None)
        对value（是一个tensor）的第一维进行随机化。
       [[1,2],            [[2,3],
        [2,3],     ==>  [1,2],
        [3,4]]             [3,4]] 

13、tf.set_random_seed(seed)
        设置产生随机数的种子。

当我们导入tensorflow包的时候，系统已经帮助我们产生了一个默认的图，它被存在_default_graph_stack中，但是我们没有权限直接进入这个图，我们需要使用tf.get_default_graph()命令来获取图。      
        
        
#%% tf.name_scopy tf.variable_scopy  diffrenent 
#name_scope 是给op_name加前缀, variable_scope是给get_variable()创建的变量的名字加前缀(and error if the varible is exit)。should not be mix used
with tf.name_scope("hildden") as scope:
    a=tf.constant(5,name='alptha')
    W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0),name='weights')
    with tf.name_scope('hildden2'):
        b=tf.Variable([0],name='b')
    print(a.name,W.name,b.name)#hildden/alptha:0 hildden/weights:0 hildden/hildden2/b:0
 
with tf.variable_scope('scope1'):
    a2=tf.constant(6,name='alptha2')
    w2=tf.get_variable('w2',3)
    w3=tf.Variable([3],name='w3')
    with tf.variable_scope('scope2'):
        b2=tf.get_variable('b2',3)
    print(a2.name,w2.name,w3.name,b2.name)#scope1_5/alptha2:0 scope1/w2:0 scope1_5/w3:0 scope1/scope2/b2:0
"""
#%%
'''
#hidden1/weights  hidden1/biases
tf.name_scope('hidden1'):
    weights=tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],stddev=1.0/ math.sqrt(float(IMAGE_PIXELS))),name='weights')
    biases=tf.Variable(tf.zeros([hidden1_units]),name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
'''

import tensorflow as tf
import numpy as np

# 生成0和1矩阵
v1 = tf.Variable(tf.zeros([3,3,3]), name="v1")#v1=tf.get_variable("v1",shape=[1],initializer=tf.constant_initializer(1.0))
v2 = tf.Variable(tf.ones([10,5]), name="v2")

#填充单值矩阵
v3 = tf.Variable(tf.fill([2,3], 9))

#常量矩阵
v4_1 = tf.constant([1, 2, 3, 4, 5, 6, 7])
v4_2 = tf.constant(-1.0, shape=[2, 3])

#生成等差数列
v6_1 = tf.linspace(10.0, 12.0, 30, name="linspace")#float32 or float64
v7_1 = tf.range(10, 20, 3)#just int32

#生成各种随机数据矩阵
v8_1 = tf.Variable(tf.random_uniform([2,4], minval=0.0, maxval=2.0, dtype=tf.float32, seed=1234, name="v8_1"))
v8_2 = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1234, name="v8_2"))
v8_3 = tf.Variable(tf.truncated_normal([2,3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1234, name="v8_3"))
v8_4 = tf.Variable(tf.random_uniform([2,3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1234, name="v8_4"))
v8_5 = tf.random_shuffle([[1,2,3],[4,5,6],[6,6,6]], seed=134, name="v8_5")

# 初始化
init_op = tf.initialize_all_variables()

# 保存变量，也可以指定保存的内容
saver = tf.train.Saver()
#saver = tf.train.Saver({"my_v2": v2})

#运行
with tf.Session() as sess:
  sess.run(init_op)
  # 输出形状和值
  print (tf.Variable.get_shape(v1))#shape
  print (sess.run(v1))#vaule
  
  # numpy保存文件
  np.save("v1.npy",sess.run(v1))#numpy save v1 as file
  test_a = np.load("v1.npy")
  print (test_a[1,2])

  #一些输出
  print (sess.run(v3))

  v5 = tf.zeros_like(sess.run(v1))

  print (sess.run(v6_1))

  print (sess.run(v7_1))

  print (sess.run(v8_5))
  
  #保存图的变量
  save_path = saver.save(sess, "/tmp/model.ckpt")
  #加载图的变量
  #saver.restore(sess, "/tmp/model.ckpt")
  print ("Model saved in file: ", save_path)
#%%

import numpy as np
import tensorflow as tf



a=tf.constant([1.0,2.0],dtype=tf.float32,name="a")
b=tf.constant([2.0,3.0],name="b")
result=a+b
sess=tf.Session()
with sess:
    print(sess.run(result)) 
    print(result.eval())
    print(result.eval(session=sess))
    
#default session
sess2=tf.Session()
with sess2.as_default():
    print(result.eval())
    
#cannot use the with as ,for it has not the __enter__ function
sess3=tf.InteractiveSession()
print(result.eval())
sess3.close()
print('\n')
#a placehold is a promise to provide a value later
c=tf.placeholder(tf.float32)
d=tf.placeholder(tf.float32)
adder_node=c+d
adder_node_triper=adder_node*3
x=tf.placeholder(tf.float32,shape=(5,5),name='x')
y=tf.matmul(x,x)
print(tf.Session().run(adder_node,feed_dict={c:4.0,d:4.5}))
print(tf.Session().run(adder_node, {c: [1,3], d: [2.0, 4]}))
print(tf.Session().run(adder_node_triper,{c:4.0,d:4.5}))
rand_array=np.random.rand(5,5)
print(tf.Session().run(y,feed_dict={x:rand_array}))

#Variables allow us to add trainable parameters to a graph.variables are not initialized
# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")

W = tf.Variable([.3],dtype=tf.float32,name='W')
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
z=tf.Variable([3.],tf.float32)

linear_model = W * x + b
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init) #initializes all the global variables
    print(sess.run(linear_model, {x:[1,2,3,4]}))
    print(W.eval())
    fixW=tf.assign(W,[.2])#reassigning the values of W
    print(fixW.eval()) #after the fixW runed ,W is changed 
    print(W.eval())



#%%numpy->tensorflow
"""
feature_column_data = [1, 2.4, 0, 9.9, 3, 120]
feature_tensor = tf.constant(feature_column_data)

"""


import numpy as np
import tensorflow as tf
a=np.zeros((3,3))
ta=tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(ta.eval())


#%% a example


import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
#%%
#%%
"""model save and restore"""
import tensorflow as tf
import numpy as np

# save to file
W = tf.Variable([[1,2,3],[4,5,6]],dtype = tf.float32,name='weight')
b = tf.Variable([[1,2,3]],dtype = tf.float32,name='biases')

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess,"tmp/save_net.ckpt")
        print ("save to path:",save_path)

#%%
import tensorflow as tf
import numpy as np
'''reset the default_graph to restore the saved ckpt'''
from tensorflow.python.framework import ops
ops.reset_default_graph()


W = tf.Variable(np.arange(6).reshape((2,3)),dtype = tf.float32,name='weight')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype = tf.float32,name='biases')

saver = tf.train.Saver()
with tf.Session() as sess:
        saver.restore(sess,"tmp/save_net.ckpt")
        print ("weights:",sess.run(W))
        print ("biases:",sess.run(b))
#%%
import requests
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
print(len(housing_data))
print(len(housing_data[0]))
#%%%线性回归
# Combining Everything Together
#----------------------------------
# This file will perform binary classification on the
# class if iris dataset. We will only predict if a flower is
# I.setosa or not.
#
# We will create a simple binary classifier by creating a line
# and running everything through a sigmoid to get a binary predictor.
# The two features we will use are pedal length and pedal width.
#
# We will use batch training, but this can be easily
# adapted to stochastic training.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Load the iris data
# iris.target = {0, 1, 2}, where '0' is setosa
# iris.data ~ [sepal.width, sepal.length, pedal.width, pedal.length]
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])    #2features

# Declare batch size
batch_size = 20

# Create graph
sess = tf.Session()

# Declare placeholders
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables A and b (0 = x1 - A*x2 + b)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Add model to graph:
# x1 - A*x2 + b
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)  
#my_output = tf.sub(x_data[0], tf.add(tf.matmul(x_data[1], A), b))

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output,labels=y_target)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Run Loop
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    #rand_x = np.transpose([iris_2d[rand_index]])
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    #rand_y = np.transpose([binary_target[rand_index]])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
        

# Visualize Results
# Pull out slope/intercept
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)

# Create fitted line
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
  ablineValues.append(slope*i+intercept)  #AX+B

# Plot the fitted line over the data
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()
#%%
'''我们的模型训练出来想给别人用，或者是我今天训练不完，明天想接着训练，怎么办？这就需要模型的保存与读取'''
import tensorflow as tf
import numpy as np
import os
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()     #tf.reset_default_graph()
#输入数据
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data)-0.5+noise

#输入层
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#隐层
W1 = tf.Variable(tf.random_normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10])+0.1)
Wx_plus_b1 = tf.matmul(xs,W1) + b1
output1 = tf.nn.relu(Wx_plus_b1)

#输出层
W2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1])+0.1)
Wx_plus_b2 = tf.matmul(output1,W2) + b2
output2 = Wx_plus_b2

#损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-output2),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#模型保存加载工具
saver = tf.train.Saver()

#判断模型保存路径是否存在，不存在就创建
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

#初始化
sess = tf.Session()
if os.path.exists('tmp/checkpoint'): #判断模型是否存在
    saver.restore(sess, 'tmp/model.ckpt') #存在就从模型中恢复变量
else:
    init = tf.global_variables_initializer() #不存在就初始化变量
    sess.run(init)

#训练
for i in range(1000):
    _,loss_value = sess.run([train_step,loss], feed_dict={xs:x_data,ys:y_data})
    if(i%50==0): #每50次保存一次模型
        save_path = saver.save(sess, 'tmp/model.ckpt') #保存模型到tmp/model.ckpt，注意一定要有一层文件夹，否则保存不成功！！！
        print("模型保存：%s 当前训练损失：%s"%(save_path, loss_value))    
        
#%%
#%%





#%%MNIST FOR ML  begninners
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(logits=tf.matmul(x, W) + b)
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))#better at:log(tf.clip_by_value(y,1e-10,10))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#%%Deep MNIST for Experts 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction=tf.equal(tf.argmax(input=y,axis=1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#Build a Multilayer Convolutional Network
def weight_variable(shape):
    initial=tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#output[i] = reduce(value[strides * i:strides * i + ksize])
 #first conv layer [m 28*28*1]->[m 28 28 1]->[m 28 28 32]->[m 14 14 32]
W_conv1 = weight_variable([5, 5, 1, 32])#filting 
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])#-1 means that not changed ,[m ,28*28*2]->[m,28,28,1]
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
 #second convolutional layer [m 14 14 32]->[m 14 14 64]->[m 7 7 64]
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
 #full-connected layer [m 7 7 64]->[m 7*7*64]->[m 1024]
h_pool2_flat=tf.reshape(h_pool2,[-1, 7*7*64])
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
 #droput reduce overfitting befor readout layer
keep_prob=tf.placeholder(dtype=tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob)
 #readout layer [m 1024]->[m 10]
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
 #train and evaluate model
cross_entropy=tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={
                x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d, training accuracy %g\n"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
print("test accuracy %g\n"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#%%%  tf.contrib.learn Quickstart 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

# Data sets   如果打开失败可手动到网站下载csv
#特征x:Sepal Length 	Sepal Width 	Petal Length 	Petal Width 	类别y:Species 
IRIS_TRAINING = "/home/keetsky/Desktop/tensorflow_learn/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
#IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "iris_training.csv")
#IRIS_TEST = os.path.join(os.path.dirname(__file__), "iris_test.csv")

IRIS_TEST = "/home/keetsky/Desktop/tensorflow_learn/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb+") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb+") as f:
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

  predictions = list(classifier.predict(input_fn=new_samples))
 #从tmp/iris_model加载的模型中进行预测
  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))

if __name__ == "__main__":
    main()

#%%
"""
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

"""

#%%Building Input Functions with tf.contrib.learn 
"""boston housing predict   mei tiao tong"""
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""DNNRegressor with custom input_fn for Housing dataset."""

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
#%%Logging and Monitoring Basics with tf.contrib.learn 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING ="/home/keetsky/Desktop/tensorflow_learn/iris_training.csv"
IRIS_TEST = "/home/keetsky/Desktop/tensorflow_learn/iris_test.csv"


    # Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)

    # Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(test_set.data,
                                                                 test_set.target,
                                                                 every_n_steps=50)


    # Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model",
                                            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
                monitors=[validation_monitor])

    # Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

    # Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))

#%%

import numpy as np
import cv2
img = cv2.imread('/home/keetsky/Desktop/tensorflow_learn/1.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

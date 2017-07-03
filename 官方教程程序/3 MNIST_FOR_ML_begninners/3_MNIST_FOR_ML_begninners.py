
# coding: utf-8

# ## MNIST FOR ML  begninners

# In[3]:

import tensorflow as tf


# ### 加载mnist数据

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ### 定义网络interface

# In[4]:

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(logits=tf.matmul(x, W) + b)
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])


# ### 定义交叉熵和梯度下降

# In[7]:

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#better at:log(tf.clip_by_value(y,1e-10,10))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# ### 变量初始化

# In[8]:

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# ### 训练

# In[12]:

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# In[ ]:




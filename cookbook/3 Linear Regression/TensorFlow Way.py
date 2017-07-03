'''
# Linear Regression: Tensorflow Way
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve linear regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris=datasets.load_iris()
x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])
# Declare batch size
batch_size = 25
# Initialize placeholders
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_=tf.placeholder(shape=[None,1], dtype=tf.float32)

#create variable for linear regression
A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
#declare model operations
y=tf.add(tf.matmul(x_data,A),b)
#declare loss functions  (1/2/m)  (y_-y)^2
loss=tf.reduce_mean(tf.square(y_- y))
#Declare optimizer
op=tf.train.GradientDescentOptimizer(0.05)
train_step=op.minimize(loss)
#initialize variables
init=tf.global_variables_initializer()
sess.run(init)
#training loop 
loss_vec=[]
for i in range(100):
    rand_index=np.random.choice(len(x_vals),size=batch_size)#随机从len(x_vals)中选取25个下标
    rand_x=np.transpose([x_vals[rand_index]])
    rand_y=np.transpose([y_vals[rand_index]])
    sess.run(train_step,feed_dict={x_data:rand_x,y_:rand_y})
    temp_loss=sess.run(loss,feed_dict={x_data:rand_x,y_:rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))
#get the optimal coefficients
[A_final]=sess.run(A) #[A_final] if not get the []  ,cannot plot best_fit_y
[b_final]=sess.run(b)
#get the best fit line
best_fit_y=[]
for i in x_vals:
    best_fit_y.append(A_final*i+b_final)

    #best_fit=(np.add(np.dot(x_vals,np.matrix(A_final)),b_final)).tolist()
#plot the result
# Plot the result
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit_y, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
#plot loss over time(steps)
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()


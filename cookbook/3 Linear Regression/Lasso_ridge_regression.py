'''
# Lasso and Ridge Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve lasso or ridge regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width

lasso regression: 
Lasso regularization(L1正则化)
loss=J(W,b)+nu*sum(|w|)/m

ridge regression:
Ridge regulariztion(L2正则化)
loss=J(W，b)+nu*sum(w^2)/m

'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

#creat graph
sess=tf.Session()
#load data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris=datasets.load_iris()
x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])
#declare bathc size
batch_size=50
#initialize placeholders
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
#create varizbele for linear regeression
A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
#declar model operations
model_output=tf.add(tf.matmul(x_data,A),b)

#declare lasso lossfunction
#lasso_param=tf.constant(1.)
#lasso_loss=tf.reduce_mean(tf.square(A))
#J_loss=tf.reduce_mean(tf.abs(y_target-model_output))
#loss=tf.expand_dims(tf.add(J_loss,tf.multiply(lasso_param,lasso_loss)),0)
#declare ridge loss function
ridge_param=tf.constant(1.)
ridge_loss=tf.reduce_mean(tf.square(A))
J_loss=tf.reduce_mean(tf.square(y_target-model_output))
loss=tf.expand_dims(tf.add(J_loss,tf.multiply(ridge_param,ridge_loss)),0)

#initialize varizbles
init=tf.global_variables_initializer()
sess.run(init)
#declare optimizer
my_opt=tf.train.GradientDescentOptimizer(0.001)
train_step=my_opt.minimize(loss)
#training loop 
loss_vec = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss[0])
    if (i+1)%300==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# Get the optimal coefficients
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# Get best fit line
best_fit = []
for i in x_vals:
  best_fit.append(slope*i+y_intercept)

# Plot the result
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()

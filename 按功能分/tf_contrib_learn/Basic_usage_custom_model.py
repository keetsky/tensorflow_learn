
#%%Basic usage  采用自带模式
import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
print(estimator.evaluate(input_fn=input_fn))
#result:
'''
INFO:tensorflow:loss = 6.75, step = 1
INFO:tensorflow:global_step/sec: 911.342
INFO:tensorflow:loss = 0.160967, step = 101 (0.110 sec)
INFO:tensorflow:global_step/sec: 984.017
INFO:tensorflow:loss = 0.0415095, step = 201 (0.102 sec)
INFO:tensorflow:global_step/sec: 1043.33
INFO:tensorflow:loss = 0.0102464, step = 301 (0.096 sec)
INFO:tensorflow:global_step/sec: 917.538
INFO:tensorflow:loss = 0.00186342, step = 401 (0.109 sec)
INFO:tensorflow:global_step/sec: 969.927
INFO:tensorflow:loss = 0.00122761, step = 501 (0.103 sec)
INFO:tensorflow:global_step/sec: 972.755
INFO:tensorflow:loss = 0.000197689, step = 601 (0.103 sec)
INFO:tensorflow:global_step/sec: 1045.47
INFO:tensorflow:loss = 7.29189e-05, step = 701 (0.096 sec)
INFO:tensorflow:global_step/sec: 987.095
INFO:tensorflow:loss = 2.64494e-05, step = 801 (0.101 sec)
INFO:tensorflow:global_step/sec: 1151.84
INFO:tensorflow:loss = 9.47965e-06, step = 901 (0.087 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /tmp/tmpwjgbmxtt/model.ckpt.    
{'loss': 1.6635287e-06, 'global_step': 1000}
'''
#%%A custom model 用户定义模型模式
'''
To define a custom model that works with tf.contrib.learn, we need to use tf.contrib.learn.Estimator. 
tf.contrib.learn.LinearRegressor is actually a sub-class of tf.contrib.learn.Estimator. 
Instead of sub-classing Estimator, 
we simply provide Estimator a function model_fn that tells tf.contrib.learn 
how it can evaluate predictions, training steps, and loss. The code is as follows:

'''
import numpy as np
import tensorflow as tf
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data set
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))
#result
'''
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpse3yxffz/model.ckpt.
INFO:tensorflow:loss = 21.1209709341, step = 1
INFO:tensorflow:global_step/sec: 1316.31
INFO:tensorflow:loss = 0.236354546961, step = 101 (0.077 sec)
INFO:tensorflow:global_step/sec: 1522.14
INFO:tensorflow:loss = 0.0124695381631, step = 201 (0.066 sec)
INFO:tensorflow:global_step/sec: 1596.43
INFO:tensorflow:loss = 0.000729889397231, step = 301 (0.063 sec)
INFO:tensorflow:global_step/sec: 1622.46
INFO:tensorflow:loss = 9.15394860924e-05, step = 401 (0.062 sec)
INFO:tensorflow:global_step/sec: 1624.89
INFO:tensorflow:loss = 4.17833296434e-06, step = 501 (0.061 sec)
INFO:tensorflow:global_step/sec: 1468.35
INFO:tensorflow:loss = 9.660222815e-07, step = 601 (0.068 sec)
INFO:tensorflow:global_step/sec: 1567.46
INFO:tensorflow:loss = 6.9949607728e-09, step = 701 (0.064 sec)
INFO:tensorflow:global_step/sec: 1669.82
INFO:tensorflow:loss = 4.0987561232e-09, step = 801 (0.060 sec)
INFO:tensorflow:global_step/sec: 1757.21
INFO:tensorflow:loss = 1.31910412433e-13, step = 901 (0.057 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /tmp/tmpse3yxffz/model.ckpt.
INFO:tensorflow:Loss for final step: 4.13292549928e-11.
INFO:tensorflow:Starting evaluation at 2017-06-15-03:11:20
INFO:tensorflow:Restoring parameters from /tmp/tmpse3yxffz/model.ckpt-1000
INFO:tensorflow:Evaluation [1/10]
INFO:tensorflow:Evaluation [2/10]
INFO:tensorflow:Evaluation [3/10]
INFO:tensorflow:Evaluation [4/10]
INFO:tensorflow:Evaluation [5/10]
INFO:tensorflow:Evaluation [6/10]
INFO:tensorflow:Evaluation [7/10]
INFO:tensorflow:Evaluation [8/10]
INFO:tensorflow:Evaluation [9/10]
INFO:tensorflow:Evaluation [10/10]
INFO:tensorflow:Finished evaluation at 2017-06-15-03:11:20
INFO:tensorflow:Saving dict for global step 1000: global_step = 1000, loss = 4.50246e-11
WARNING:tensorflow:Skipping summary for global_step, must be a float or np.float32.
{'loss': 4.5024595e-11, 'global_step': 1000}

'''

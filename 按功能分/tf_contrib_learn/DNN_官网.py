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
'''
.
.
.
INFO:tensorflow:loss = 39.5997, step = 4701 (0.109 sec)
INFO:tensorflow:global_step/sec: 918.341
INFO:tensorflow:loss = 36.3882, step = 4801 (0.109 sec)
INFO:tensorflow:global_step/sec: 871.401
INFO:tensorflow:loss = 35.9476, step = 4901 (0.115 sec)
INFO:tensorflow:Saving checkpoints for 5000 into /tmp/boston_model/model.ckpt.
predictions: [34.335186, 21.191118, 24.503973, 33.888966, 16.076784, 21.190222]
'''

############################################################
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

'''

INFO:tensorflow:Validation (step 1550): loss = 0.0617481, accuracy = 0.966667, global_step = 1544
INFO:tensorflow:global_step/sec: 216.255
INFO:tensorflow:loss = 0.0415786, step = 1601 (0.462 sec)
INFO:tensorflow:global_step/sec: 596.989
INFO:tensorflow:loss = 0.04094, step = 1701 (0.168 sec)
INFO:tensorflow:global_step/sec: 684.009
INFO:tensorflow:loss = 0.0401093, step = 1801 (0.146 sec)
INFO:tensorflow:global_step/sec: 701.993
INFO:tensorflow:loss = 0.0395429, step = 1901 (0.142 sec)
INFO:tensorflow:Saving checkpoints for 2000 into /tmp/iris_model/model.ckpt.
INFO:tensorflow:Saving dict for global step 2000: accuracy = 0.966667, global_step = 2000, loss = 0.0649375
INFO:tensorflow:Restoring parameters from /tmp/iris_model/model.ckpt-2000
Predictions: [1, 2]
'''




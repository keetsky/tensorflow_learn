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

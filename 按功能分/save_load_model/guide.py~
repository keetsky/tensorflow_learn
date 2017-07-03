'''
保存格式：
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'

'''

########################
#save
...
# Create a saver.
saver = tf.train.Saver(...variables...)
'''
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})

'''     
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.Session()
for step in xrange(1000000):
    sess.run(..training_op..)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)

#################################
#Exporting a Complete Model to MetaGraph
#Export the default running graph:
'''
# Build the model
...
with tf.Session() as sess:
  # Use the model
  ...
# Export the model to /tmp/my-model.meta.
meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')
'''
#Export the default running graph and only a subset of the collections.
'''
meta_graph_def = tf.train.export_meta_graph(
    filename='/tmp/my-model.meta',
    collection_list=["input_tensor", "output_tensor"])
'''


#Import a MetaGraph
#Import and continue training without building the model from scratch
'''
...
# Create a saver.
saver = tf.train.Saver(...variables...)
# Remember the training_op we want to run by adding it to a collection.
tf.add_to_collection('train_op', train_op)
sess = tf.Session()
for step in xrange(1000000):
    sess.run(train_op)
    if step % 1000 == 0:
        # Saves checkpoint, which by default also exports a meta_graph
        # named 'my-model-global_step.meta'.
        saver.save(sess, 'my-model', global_step=step)
'''
# training from this saved meta_graph without building the model from scratch.
'''
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # tf.get_collection() returns a list. In this example we only want the
  # first one.
  train_op = tf.get_collection('train_op')[0]
  for step in xrange(1000000):
    sess.run(train_op)
'''
###################
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

###################################################################3
###程序例子
##Import and extend the graph
# Creates an inference graph.
# Hidden 1
images = tf.constant(1.2, tf.float32, shape=[100, 28])
with tf.name_scope("hidden1"):
  weights = tf.Variable(
      tf.truncated_normal([28, 128],
                          stddev=1.0 / math.sqrt(float(28))),
      name="weights")
  biases = tf.Variable(tf.zeros([128]),
                       name="biases")
  hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
# Hidden 2
with tf.name_scope("hidden2"):
  weights = tf.Variable(
      tf.truncated_normal([128, 32],
                          stddev=1.0 / math.sqrt(float(128))),
      name="weights")
  biases = tf.Variable(tf.zeros([32]),
                       name="biases")
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
# Linear
with tf.name_scope("softmax_linear"):
  weights = tf.Variable(
      tf.truncated_normal([32, 10],
                          stddev=1.0 / math.sqrt(float(32))),
      name="weights")
  biases = tf.Variable(tf.zeros([10]),
                       name="biases")
  logits = tf.matmul(hidden2, weights) + biases
  tf.add_to_collection("logits", logits)

init_all_op = tf.global_variables_initializer()

with tf.Session() as sess:
  # Initializes all the variables.
  sess.run(init_all_op)
  # Runs to logit.
  sess.run(logits)
  # Creates a saver.
  saver0 = tf.train.Saver()
  saver0.save(sess, 'my-save-dir/my-model-10000')
  # Generates MetaGraphDef.
  saver0.export_meta_graph('my-save-dir/my-model-10000.meta')



## import it and extend it to a training graph.
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # Addes loss and train.
  labels = tf.constant(0, tf.int32, shape=[100], name="labels")
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat([indices, labels], 1)
  onehot_labels = tf.sparse_to_dense(
      concated, tf.stack([batch_size, 10]), 1.0, 0.0)
  logits = tf.get_collection("logits")[0]
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=onehot_labels, logits=logits, name="xentropy")
  loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")

  tf.summary.scalar('loss', loss)
  # Creates the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(0.01)

  # Runs train_op.
  train_op = optimizer.minimize(loss)
  sess.run(train_op)
##其他额外
#Import a graph with preset devices.
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta',
      clear_devices=True)
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  ...


#Import within the default graph.
meta_graph_def = tf.train.export_meta_graph()
...
tf.reset_default_graph()
...
tf.train.import_meta_graph(meta_graph_def)
...
#Retrieve Hyper Parameters
filename = ".".join([tf.train.latest_checkpoint(train_dir), "meta"])
tf.train.import_meta_graph(filename)
hparams = tf.get_collection("hparams")





























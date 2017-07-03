import tensorflow as tf
import numpy as np
from matplotlib.pyplot as plt
from tensorflow.python.framework import ops
#ops.reset_default_graph()
tf.reset_default_graph()
sess = tf.Session()#sess = tf.InteractiveSession()

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.truncated_normal([in_size, out_size]), stddev=0.1,name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
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

l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)


# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct_prediction=tf.equal(tf.argmax(y_prdict,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

writer = tf.summary.FileWriter("logs/", sess.graph)
merged = tf.summary.merge_all()#merged = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(train_dir,graph_def=sess.graph_def)
loss_vec=[]
train_acc = []
vali_acc=[]
tf.global_variables_initializer().run()
#with tf.Session() as sess:
for step in range(1000):
    #train
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train}) 
    train_feed={xs: x_train, ys: y_train} 
    valid_feed={xs: x_validate, ys: y_validate}
    if step % 50 ==0：
        acc_train=sess.run(accuracy, feed_dict=train_feed)
        train_acc.append(acc_train)
	vali_acc.append(sess.run(accuracy, feed_dict=valid_feed))
	loss_vec.append(loss,feed_dict=train_feed)    
       	summary_str = sess.run(merged, feed_dict=train_feed)
       	summary_writer.add_summary(summary_str, step)
test_acc=sess.run(accuracy, feed_dict={xs: x_test, ys:y_test})
print("test accuracy: %g\n"%test_acc)

plt.plot(loss_vec, 'r--', label='Loss')
plt.title(' Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

plt.plot(train_acc, 'r--', label='train_acc')
plt.plot(valid_acc, 'k-', label='valid_acc')
plt.title(' train_valid_acc per Generation')
plt.xlabel('Generation')
plt.ylabel('train_valid_acc')
plt.legend(loc='upper right')
plt.show()













#http://www.360doc.com/content/17/0321/10/10408243_638692495.shtml



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()
tf.set_random_seed(1)

#input data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#hyperparameters
lr=0.001               #learning rate
training_iters=100000  #train step
batch_size=128         #
n_inputs=28            #MNIST data input(img shape:28*28)
n_steps=28             #time steps
n_hidden_unists=128    #neurons in hidden layer
n_classes=10           #MNIST classes (0-9 digists)

x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

weights={
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unists])),
        #shpe(28,128)
        'out':tf.Variable(tf.random_normal([n_hidden_unists,n_classes]))
        #shape(128,10)
        }
        
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unists,])),
        #shape(128,)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
        #shape(10,)
        }

def RNN(X,weights,biases):
    X=tf.reshape(X,[-1,n_inputs])#3->2wei,X->(128*28,28)
    X_in=tf.matmul(X,weights['in'])+biases['in']
    #X_in->(128,28,128)  
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_unists])
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_unists,forget_bias=1.0,state_is_tuple=True)
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
    #results=tf.matmul(final_state[1],weights['out'])+biases['out']
    outputs=tf.unstack(tf.transpose(outputs,[1,0,2]))
    results=tf.matmul(outputs[-1],weights['out'])+biases['out']
    return results

pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=0
    while step*batch_size < training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
        if step % 20 ==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step+=1

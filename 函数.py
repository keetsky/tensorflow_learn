

###################

数据类型：
	tensorflow_type  python_type
	DT_FLOAT	tf.float32				
	DT_INT16	tf.int16
	DT_INT32	tf.int32
	DT_INT64	tf.int64
	DT_STRING	tf.string
	DT_BOOL		tf.bool
	DT_UINT8	tf.uint8
	DT_UINT16	tf.uint16
(4D = [batch size, width, height, channels])


1. 将pythons数据化为图
   a=np.zeros((3,3))
   ta=tf.convert_to_tensor(a)



2. Graph 设置默认图,构建外部运算图
    sess = tf.InteractiveSession()
    with tf.Graph().as_default():
  #重设
    tf.reset_default_graph()
  #
    from tensorflow.python.framework import ops
    ops.reset_default_graph()

sess=tf.Session()
3. 稀疏矩阵定义，分别在a[0][1]=6,a[2][4]=0.5
    a = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])

3. 基本
    tf.constant() 				#定义常量
    tf.zeros([2,3])				#0矩阵
    tf.ones()					#1矩阵
    tf.fill([2,3],33)				#填充
    tf.zeros_like(a)				#定义0矩阵，shape与矩阵a相同
    tf.ones_like(a)				#定义1矩阵，shape与矩阵a相同
    tf.identity(a)				#定义单位矩阵，shape与矩阵a相同

    tf.linespace(start=0,stop=1,num=3)		#定义一个系列，数字大小范围为0-1，个数为3
    tf.range(start=6,limit=15,delta=3)		#定义一个系列，数字大小范围为6-15，步大小为3
  #随机数
    tf.random_uniform([row_dim,col_dim],minval=0,maxval=1)		#生成0-1范围内的随机数服从均匀分布
    tf.random_normal([row_dim,col_dim],mean=0.0,stddev=1.0)		#生成均值为0.0，标准差1.0的高斯分布
    tf.truncated_normal([row_dim, col_dim],mean=0.0, stddev=1.0)	#生成均值为0.0，标准差1.0的截断高斯发布
    tf.random_shuffle(input_tensor)		#对输入张量第一维进行随机化（行）
    tf.random_crop(input_tensor,crop_size)	#随机修剪一个张量到指定大小Cropped_image=tf.random_crop(my_image,[height/2,width/2,3])
    tf.set_random_seed()			#sets the random seed		

  #变量
    tf.Variable(tf.zeros([2,3]))			#定义变量	
    initialize_op=tf.global_variables_initializer()	#定义初始化变量or  tf.initialize_all_variables()
    sess.run(initialize_op)				#初始化变量or sess.run(tf.initialize_all_variables())
    variables=tf.global_variables()			#获取程序中的变量==tf.all_variables()
    tf.trainable_variables()#返回的是需要训练的变量列表,当对于变量的参数trainable=False时将不再返回此变量
    tf.assign()    
	#a=tf.Variable(0.0)
	#a_update=tf.assign(a,tf.constant(4.0))
    (2). 设置变量域名 如：hidden1/biases
	with tf.name_scope('hidden1') as scope:
	    biases=tf.Variable(0.0,name='biases')
  #placeholders占位符 先定义空变量，最后进行初始化feed
    x=tf.placeholder(tf.float32,shap=[2,2])
    y=tf.identity(x)
    x_vals=np.random.rand(2,2)
    sess.run(y,feed_dict={x:x_vals})
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
4. 逻辑运算
    tf.equal()

4. 算数运算
    tf.add()
    tf.add_n()				#把所有输入的tensor按照元素相加
    tf.substract()
    tf.multiply()			#元素相乘
    tf.scalar_mul()			#标量乘与向量
     
    tf.div(3,4)				#普通整除法:0
    tf.truediv(3,4)			#真实除法:0.75
    tf.floordiv(3.0,4.0)		#地板除法:0.0
    tf.mod(22.0,5.0)			#取余:2.0
    tf.cross([1.,0.,0.],[0.,1.0,0.0])	#:[0.,0.,1.0]?
    tf.abs(x)		#|x|
    tf.ceil()
    tf.cos()
    tf.exp()
    tf.floor()
    tf.inv()		#y=1/x
    tf.log()
    tf.maximum()
    tf.minimun()
    tf.neg(x)		#y=-x
    tf.pow()		#x^y
    tf.round()
    tf.rsqrt()		#y=1/x
    tf.sign()
    tf.sin()
    tf.sqrt()
    tf.square()		#y=x^2
    tf.digamma()
    tf.sigmoid()
    tf.tan()
    tf.atan()
  #复数计算
    tf.complex(1,2)	#创建复数 1+2j
    tf.complex_abs()	#计算复数大小sqrt(real^2+imag^2)
    tf.conj()		#计算共轭复数
    tf.imag()		#提取虚部
    tf.real()		#提取实部
    tf.fft()		#计算一维的离散傅里叶变换
5. 矩阵运算
    tf.shape()
    a.get_shape()   #a,get_shape().as_list()      a.get_shape()[1]
    tf.size()
    tf.rank()
    tf.reshape()
    a.set_shape()   #功能与tf.reshape相同   
    tf.squeeze()	#消除维度为1的部分,t[1,2,1,3,1,1],squeeze(t)->[2,3],squeeze(t,[2,4])->[1,2,3,1]将第2,4维降除
    tf.expand_dims()	#insert a dimension to a tensor,假定t2维度为[2,3,5],扩充第0维expand_dims(t2,0)->[1,2,3,5],expand_dims(t2,1)->[2,1,3,5],expand_dims(t2,-1)->[2,3,5,1]
    tf.slice()
    tf.split()
    tf.tile()		#create a new tesor replicating a tensor multiple times
    tf.concat()		#连接多通道， concatenate tensors in oen dimension,参考np.concatenate,tf.concat([t1,t2],axis=3),连接通道3，常用于卷积神经网络通道连接
    tf.reverse()
    tf.gather()		#to collect portions according to an index
    tf.stack()		#tf.stack([[1,4],[2,5]])== [[1,4],[2,5]],tf.stack([[1,4],[2,5]],axis=1)==[[1,2],[4,5]]   
    	   



    tf.diag([1.,2.，3.])	#创建对角矩阵
    tf.trace()			#对角值之和
    tf.matmul() 		#矩阵乘法
    tf.transpose()		#矩阵转置
    tf.matrix_determinant(D)	#矩阵行列式
    f.matrix_inverse(D)		#矩阵逆矩阵
    tf.cholesky()		#矩阵cholesky分解，把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解A=LL^T
    tf.self_adjoint_eig(D)	#矩阵特征值和特征向量求解，第一行为特征值，其他行列为特征向量
    tf.squeeze(input_matrix)	#去掉维度为1的
    tf.matrix_solve(A，Y)	#线性方程求解
    y=tf.add(tf.matmul(x_data, A), b)#y=x*A+b
  #数据类型转换
    tf.string_to_number()
    tf.to_double()			#->float64
    tf.to_float()			#->float32
    tf.to_int32()
    tf.to_int64()
    tf.cast(x, dtype, name=None) 	#将x或者x.values转换为dtype
  #reduce规约运算
    tf.reduce_sum()#computes the sum of elements along one dimension
    tf.reduce_prod() #computes the product of elements along one dimension
    tf.reduce_min()#computes the mimimum of elements along one dimension
    tf.reduce_max()
    tf.reduce_mean()#computes the mean of elements along one dimension
    tf.reduce_all()#对tensor中各个元素求逻辑’与
    tf.reduce_any()#对tensor中各个元素求逻辑或
    tf.accumulate_n()#计算一系列张量的和
  #arg索引
    tf.argmin()#return the index of the element with the minimum value along tensor dimension
    tf.argmax()#数据最大值所在的索引值。由于标签向量是由0,1组成,因此最大值1所在的索引位置就是类别标签
  #稀疏矩阵tensor创建
    spare_tensor=tf.SparseTensor(indices=[[0,1],[1,2]],value=[6,0.5],dense_shape=[2,3])#[[0,6,0],[0,0,0.5]]



    tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')#填充，在第2,3维边界以0扩充3行数列
6. 集合collection
   tf.add_to_collection("logits", logits)#将logits放入名为logits的集合中			
   logits2= tf.get_collection("logits")[0]



6. 激活函数
    tf.nn.relu()   	 #max(0,x)
    tf.nn.relu6()   	 #:min(max(0,x),6)
    tf.nn.sigmoid() 	 #：1/(1+exp(-x))      值域[0,1]
    tf.nn.tanh()	 #：(exp(x)-exp(-x))/(exp(x)+exp(-x)）     值域 [-1,1]
    tf.nn.softsign()     #：x/(abs(x)+1)
    tf.nn.softplus()     #:log(exp(x)+1)
    tf.nn.elu()		 #：x<0 ? exp(x)+1:x 
    tf.nn.softmax()	 #exp(x_i)/sum(exp(x_j))
7. 损失函数
    1.L2 norm loss
	l2_y_vals = tf.square(target - y_vals)
        loss = tf.reduce_mean(l2_y_vals)
    2.L1 norm loss
	#l1_y_vals = tf.abs(target - x_vals)
	loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))
    3.Pseudo-huber loss
	delta1 = tf.constant(0.25)
	phuber1_y_vals = tf.mul(tf.square(delta1), tf.sqrt(1. +tf.square((target - x_vals)/delta1)) - 1.)
    4.classfication loss

    5.hinge loss
	hinge_y_vals = tf.maximum(0., 1. - tf.mul(target, x_vals))l
    6.cross-entropy 交叉熵 #− tf.reduce_sum(y_*tf.log(y))
	xentropy_y_vals = - tf.mul(target, tf.log(x_vals)) - tf.mul((1. -target), tf.log(1. - x_vals))
    	loss=tf.reduce_mean(xentropy_vals)
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y), reduction_indices=[1]))

    7.sigmoid cross entropy loss
	xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(x_vals, targets)
    8.weighted cross entropy
	eight = tf.constant(0.5)
	xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
    9.softmax cross-entropy
	softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits, target_dist)
    10.sparse softmax cross-entropy
	sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_logits, sparse_target_dist)
    11.deming regression loss#对于线性回归d=|y-(x*A+b)|/(A^2+1)^(1/2)
	demming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
	demming_denominator = tf.sqrt(tf.add(tf.square(A),1))
	loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

  #保存loss
	loss_vec=[]
	for i in range(1000):
	loss_vec.append(sess.run(loss,feed_dict={.......}))
8. 反向传播
    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    train_step = my_opt.minimize(loss)
   
    针对标准的梯度下降容易再平区域学习缓慢的问题可采用：MomentumOptimizer()
    优化每个变量：对变化小的变量采取大的steps,大变化的变量采取小的steps:
	tf.train.AdagradOptimizer()
		AdadeltaOptimizer()
		AdamOptimizer()
  (2). train_op 实现一次完成多个操作
     train_op=tf.group(train_step,averages_op)#常用于滑动平均参数的加入
     #or
     with tf.contol_dependencies([train_step,averages_op]):
	train_op=tf.no_op(name='train')
     #也可写成
     with tf.contol_dependencies([train_step]):
	train_op=tf.group(averages_op)

     sess.run(train_op,feed_dict=....)	
      
12. 数据处理
    (1). 随机batch
	rand_index = np.random.choice(len(x_vals), size=batch_size)##随机从0-len(x_vals)中选取batch_size个下标组成rand_index=array([......])
  	rand_x = np.transpose([x_vals[rand_index]])
    	rand_y = np.transpose([y_vals[rand_index]])
    (2). 数据划分
	# Split data into train/test = 80%/20%
	train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	x_vals_train = x_vals[train_indices]
	x_vals_test = x_vals[test_indices]
	y_vals_train = y_vals[train_indices]
	y_vals_test = y_vals[test_indices]
    (3) 标准化
	# Normalize by column (min-max norm)
	def normalize_cols(m):
    	    col_max = m.max(axis=0)
   	    col_min = m.min(axis=0)
   	    return (m-col_min) / (col_max - col_min)
    
	x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))#用一个很大的数表示无穷大数,一个很小的数表示无穷小数
	x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))
13. 正则化
    (1). Lasso regularization(L1正则化), loss=J(W,b)+nu*sum(|w|)/m
	lasso_param = tf.constant(1.)
	l1_a_loss = tf.reduce_mean(tf.abs(w))
	loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_ - y)), tf.mul(lasso_param, l1_a_loss)), 0)
    (2).Ridge regulariztion(L2正则化)，loss=J(W，b)+nu*sum(w^2)/m
	ridge_param = tf.constant(1.)
	l2_a_loss = tf.reduce_mean(tf.square(w))
	loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_ - y)), tf.mul(ridge_param, l2_a_loss)), 0)
    (3).lastic Net regylariztion(L1+L2混合正则化，弹性网络)，loss=J(W,b)+nu1*sum(|w|)/m+nu2*sum(w^2)/m
	elastic_param1 = tf.constant(1.)
	elastic_param2 = tf.constant(1.)
	l1_a_loss = tf.reduce_mean(tf.abs(A))
	l2_a_loss = tf.reduce_mean(tf.square(A))
	e1_term = tf.multiply(elastic_param1, l1_a_loss)
	e2_term = tf.multiply(elastic_param2, l2_a_loss)
	loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_ - y)), e1_term), e2_term), 0)
14. 其他优化
     (1). tf.nn.dropout() #减少过拟合对当前层优化
		keep_prob = tf.placeholder("float")
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#训练时设百分比如:0.5，测试时设为全百分比：1.0
	

18. 卷积和池化
	tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='SAME')#;input:[m,28,28,3];W:[5,5,3,32]SAME:添加全0填充,VALID:不填充.输出:[m,28,28,32]
	tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#input:[m,28,28,32],输出：[m,14,14,32]
        net=tf.reshape(net,[-1,net.get_shape[1]*net.get_shape[2]*net.get_shape[3]])
        h_fc1=tf.nn.relu(tf.matmul(net,W_fc1)+b_fc1)
20. 预测和评估
        correct_prediction=tf.equal(tf.argmax(y_prdict,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#[True False True True]->[1. 0. 1. 1.]->(1.+0.+1.+1.)/4.=0.75

     ##
	correct = tf.nn.in_top_k(logits, labels, 1)
	tf.reduce_sum(tf.cast(correct, tf.int32))	

     ##保存
	train_feed={xs: x_train, ys: y_train} 
    	valid_feed={xs: x_validate, ys: y_validate}
	train_acc = []
	valid_acc = []
	for i in range(1000):
	    acc_train=sess.run(accuracy, feed_dict=train_feed)
	    train_acc.append(acc_train) 
	    vali_acc.append(sess.run(accuracy, feed_dict=valid_feed))
51. Enabling Logging 
    #log messages:DEBUG INFO WARN FATAL,tensorflow默认为WARN，当需要跟踪训练模型则需要调整到INFO
    tf.logging.set_verbosity(tf.logging.INFO)
	#INFO:tensorflow:loss = 34.1165, step = 13301 (0.130 sec)
	#INFO:tensorflow:global_step/sec: 760.906
98. 可视化tensorboard
    tf.summary.histogram('/biases', biases) #变量w,b的可视化
    tf.summary.scalar('loss', loss)#标量loss的可视化tf.scalar_summary(loss.op.name, loss)
    writer = tf.summary.FileWriter("/path/to/logs/", sess.graph)
    merged = tf.summary.merge_all()#merged = tf.merge_all_summaries()
    #summary_writer = tf.train.SummaryWriter(train_dir,graph_def=sess.graph_def)
    for step in range(1000):
	if step % 50 ==0：
    		summary_str = sess.run(merged, feed_dict=feed_dict)
    		summary_writer.add_summary(summary_str, step)
    #运行
      tensorboard --logdir=/path/to/logs
99. 保存和加载结构和变量MetaGraph
    saver=tf.train.Saver()#saver = tf.train.Saver( tf.global_variables(), max_to_keep=None)
    tf.train.export_meta_graph(filename='/tmp/my-model.meta',collection_list=["input_tensor", "output_tensor"])#导出可选图(in/output_tensor)
    saver.save(sess, 'my-model', global_step=step)
    saver.export_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')




100. 画图
    from matplotlib.pyplot as plt

    (1).线性回归,输入和输出都是一维
	plt.plot(x_vals,y_vals,"o",label="Data")#原始数据x,y
	plt.plot(x_vals,best_fit_y,'r-',label="Best fit line",linewidth=3)#线性回归的直线
	plt.legend(loc='upper letf')
	plt.xlabel('Pedal Width')
	plt.ylabel('Sepal Length')
	plt.show()

    (7). plot loss over steps
	plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
	plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
	plt.title('L1 and L2 Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('L1_L2 Loss')
	plt.legend(loc='upper right')
	plt.show()



101. tf.contrib.learn

102. tf.contrib.slim   主要用于训练CNN的API#https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim   https://github.com/tensorflow/models/tree/master/slim#Pretrained
    import tensoflow.contrib.slim as slim
  (1). slim.arg_scope([slim.conv2d,slim.max_pool2d],stride=1,padding='VALID'),对[]里的函数自动默认赋值
       slim.conv2d   slim.max_pool2d   slim.avg_pool2d slim.full_connected  slim.l2_regularizer  slim.softmax 
  (2). weights=slim.variable('weights',shape=[10,10,3,3],initializer=tf.truncated_normal_initializer(stddev=0.1),regularizer=slim.l2_regularizer(0.05),device='/CPU:0')
  (3). net=slim.conv2d(input, 128, [3, 3], stride=2,padding='VALID', scope='conv1_1')#卷积核为3x3x128 
  (4). net=slim.max_pool2d(net, [2, 2], stride=2,padding='VALID',scope='pool2')#池化核2x2 
       slim.avg_pool2d()
  (5). x = slim.fully_connected(x, 32,activation_fn=nn.relu,trainable=True,， scope='fc/fc_1')#全连接层FC    
  (6). with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                      weights_regularizer=slim.l2_regularizer(0.0005)):#如果conv2d函数未定义这2参数，使用设置参数
           net = slim.conv2d(inputs, 64, [11, 11], scope='conv1')
           net = slim.conv2d(net, 128, [11, 11], padding='VALID', scope='conv2')
           net = slim.conv2d(net, 256, [11, 11], scope='conv3')
  (7). net = slim.dropout(net, keep_prob=0.8, is_training=is_training,scope='Dropout_1b')


111. RNN
    (1). LSTM
	lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_unists,forget_bias=1.0,state_is_tuple=True)
  	init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    	outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
    	#results=tf.matmul(final_state[1],weights['out'])+biases['out']
    	outputs=tf.unstack(tf.transpose(outputs,[1,0,2]))
    	results=tf.add(tf.matmul(outputs[-1],weights['out']),biases['out'])



120. device GPU/CPU
	c = []
	for d in ['/gpu:2', '/gpu:3']:
 	    with tf.device(d):
  	    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
   	    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
   	    c.append(tf.matmul(a, b))
	with tf.device('/cpu:0'):
  	    sum = tf.add_n(c)
        #allow_soft_placement to True使得在上述device不存在下自动选择其他device
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
	print(sess.run(sum))
  (2). 多块GPU，tensorflow在训练时默认占用所有GPU的显存， 希望指定使用特定某块GPU。 
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"#指定第二块,或再shell中输入export CUDA_VISIBLE_DEVICES=1  
'''
	Environment Variable Syntax      Results
	CUDA_VISIBLE_DEVICES=1           Only device 1 will be seen
	CUDA_VISIBLE_DEVICES=0,1         Devices 0 and 1 will be visible
	CUDA_VISIBLE_DEVICES="0,1"       Same as above, quotation marks are optional
	CUDA_VISIBLE_DEVICES=0,2,3       Devices 0, 2, 3 will be visible; device 1 is masked
	CUDA_VISIBLE_DEVICES=""          No GPU will be visible
'''
  (3). 对所有GPU使用百分比限度
    # 假如有12GB的显存并使用其中的4GB:  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
  (4).  当allow_growth设置为True时，分配器将不会指定所有的GPU内存，而是根据需求增长 
	config = tf.ConfigProto()  
	config.gpu_options.allow_growth=True  
	sess = tf.Session(config=config) 
#%%########################################################################
1. 数据集
    (1). MNIST数据集
	import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)#60000个训练数集：mnist.train，1000个测试数集：mnist.test;图片：xs=mnist.train.images,图片像素：28x28，像素值介于0和站之间，图片数集矩阵：[60000,784];标签：ys=mnist.train.labels，介于0-9之间的数字，标签数集：[60000,10],标签1：[0,1,0,0,0,0,0,0,0,0]
	batch = mnist.train.next_batch(50)
	feed_dict={x: batch[0], y_: batch[1]}
	#####
	data_sets = input_data.read_data_sets(FLAGS.train_dir , FLAGS.
fake_data)
	images_placeholder = tf.placeholder(tf.float32 , shape=(batch_size ,
IMAGE_PIXELS))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    (2). IRIS数据集 #特征x:Sepal Length 	Sepal Width 	Petal Length 	Petal Width 	类别y:Species 
	IRIS_TRAINING = "/home/keetsky/Desktop/tensorflow_learn/iris_training.csv"
	IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
	#IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "iris_training.csv")
	#IRIS_TEST = os.path.join(os.path.dirname(__file__), "iris_test.csv")

	IRIS_TEST = "/home/keetsky/Desktop/tensorflow_learn/iris_test.csv"
	IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
	if not os.path.exists(IRIS_TRAINING):
	    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
            with open(IRIS_TRAINING, "wb+") as f:
      	    f.write(raw)
 	if not os.path.exists(IRIS_TEST):
    	    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    	    with open(IRIS_TEST, "wb+") as f:
            f.write(raw)
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,target_dtype=np.int,features_dtype=np.float32)#特征少时



	#or
	  from sklearn import datasets
	  iris = datasets.load_iris()
	  x_vals = np.array([x[0:3] for x in iris.data])
	  y_vals = np.array([x[3] for x in iris.data])
	  # Split data into train/test = 80%/20%
	  train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	  test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	  x_vals_train = x_vals[train_indices]
	  x_vals_test = x_vals[test_indices]
	  y_vals_train = y_vals[train_indices]
	  y_vals_test = y_vals[test_indices]
    (3). BOSTON 数据集
	training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)
	test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)
	prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,skiprows=1, names=COLUMNS)







############################################################################
	
1. dropout #为了减少过拟合,我们在输出层之前加入 dropout,训练过程中启用 dropout ,在测试过程中关闭 dropout
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:
0.5})
	accuracy.eval(feed_dict={
x: mnist.test.images , y_: mnist.test.labels , keep_prob: 1.0})
	
2.  变量初始化
	weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units],stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),

3.  tf.one_hot#比如将数字[1,9]->[[0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1]]
    	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
4.  np.transpose tf.transpose 多维
        a=np.array([[[1,2],[3,4],[2,2]],[[5,6],[7,8],[9,8]]])
  	a.shape   #(2,3,2)
	np.transpose(a)==np.transpose(a,[2,1,0])#数字表第几维,array([[[1,5],[3,7],[2,9]],[[2,6],[4,8],[2,8]]])
	np.transpose(a,[0,2,1])#array([[[1,3,2],[2,4,2]],[[5,7,9],[6,8,8]]])对每个行块转置
        np.transpose(a,[2,0,1])
	
5. dropout 优化#训练时,最后全连接层前
        keep_prob=tf.placeholder(dtype=tf.float32)
	h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob)









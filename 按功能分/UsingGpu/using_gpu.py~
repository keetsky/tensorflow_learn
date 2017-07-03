'''

    "/cpu:0": The CPU of your machine.
    "/gpu:0": The GPU of your machine, if you have one.
    "/gpu:1": The second GPU of your machine, etc.


'''
#%Logging Device placement
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
'''
b: /job:localhost/replica:0/task:0/gpu:0
a: /job:localhost/replica:0/task:0/gpu:0
MatMul: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]

'''
#%Manual device placement
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
'''
b: /job:localhost/replica:0/task:0/cpu:0		#CPU
a: /job:localhost/replica:0/task:0/cpu:0		#CPU
MatMul: /job:localhost/replica:0/task:0/gpu:0           #GPU
[[ 22.  28.]
 [ 49.  64.]]
'''

#%Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
#设定使用40%GPU MEMORY
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)


#%Using a single GPU on a multi-GPU system
# 在特定GPU上运行，如果GPU不存在将报错
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
#set allow_soft_placement to True使得在GPU2不存在下自动选择其他device
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
print(sess.run(c))


#%Using multiple GPUs
# Creates a graph.
c = []
for d in ['/gpu:2', '/gpu:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
'''
Const_3: /job:localhost/replica:0/task:0/gpu:3
Const_2: /job:localhost/replica:0/task:0/gpu:3
MatMul_1: /job:localhost/replica:0/task:0/gpu:3
Const_1: /job:localhost/replica:0/task:0/gpu:2
Const: /job:localhost/replica:0/task:0/gpu:2
MatMul: /job:localhost/replica:0/task:0/gpu:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]
'''


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









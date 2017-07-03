#https://docs.scipy.org/doc/numpy/reference/routines.io.html

###################
#%##%文件存取#
#Numpy可以将数组保存至二进制文件、文本文件，同时支持将多个数组保存至一个文件中
1. np.tofile() np.fromfile() 
    os.chdir("d:\\")
    a=np.arange(0.12).reshape(3,4)
    a.tofile("a.bin")
    a_=np.fromfile("a.bin",dtype=np.int32)#数据默认为行1X12的矩阵;从文件中加载数组，错误的dtype会导致错误的结果
    a_reshape(3,4)
2. np.save np.load np.savez np.savez_compressed
    #load()和save()用Numpy专用的二进制格式保存数据,保留元素类型和形状等，savez()提供了将多个数组存储至一个文件的能力
    np.save("a.npy",a.reshape(3，4))
    c=np.load("a.npy")  #c为3X4矩阵，矩阵形状不变
    b=np.arange([0,1,0,0])
    np.savez("result.npz",a,b,sin_c=c)#使用sin_c重命名数组c,按字典类型存储默认键为array_0,array_1 ...，格式为.npz
    r=np.load("result.npz")#加载一次即可
    r["arr_0"]#a
    r["arr_1"]#b
    r["sin_c"]#c
    r[1,:]#第二行数据
3. np.savetxt np.loadtxt 
    np.savetxt("a.txt",a)
    np.loadtxt("a.txt")#a

###################
#Pickle  CPickle 将任意一个Python对象转换成一系统字节的这个操作过程叫做串行化对象
#CPickle是用C来实现的，它的速度要比pickle快好多倍，一般建议如果电脑中只要有CPickle的话都应该使用它
'''
pickle.dump(obj, file, [,protocol])
　　注解：将对象obj保存到文件file中去。
　　　　　protocol为序列化使用的协议版本，0：ASCII协议，所序列化的对象使用可打印的ASCII码表示；1：老式的二进制协议；2：2.3版本引入的新二进制协议，较以前的更高效。其中协议0和1兼容老版本的python。protocol默认值为0。
　　　　　file：对象保存到的类文件对象。file必须有write()接口， file可以是一个以'w'方式打开的文件或者一个StringIO对象或者其他任何实现write()接口的对象。如果protocol>=1，文件对象需要是二进制模式打开的。

　　pickle.load(file)
　　注解：从file中读取一个字符串，并将它重构为原来的python对象。
　　file:类文件对象，有read()和readline()接口。
'''
try:
    import cPickle as pickle
except:
    import pickle
info=[1,2,3,'abc','nihoa']
data1=pickle.dump(info)#
data2=pickle.load(data1)
f1=file('temp.pkl','wb')
pickle.dump(info,f1,True)
f1.close()
f2=file('temp.pkl','rb')
info2=pickle.load(f2)


python2:
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
python3:
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
	#d_decoded={}
        #for k,v in dict.items():
	    #d_decoded[k.decode('utf8')]=v   #二进制文件读取文本，需要解码
        #dict=d_decoded
    return dict
    
########################
#CSV
import csv
'''
No.,Name,Age,Score
1,Apple,12,98
2,Ben,13,97
3,Celia,14,96
4,Dave,15,95
'''
1. with open('A.csv','rb',encoding='utf-8') as f:
	reader=csv.reader(f)
	data=[row for row in reader]#一次性读取完毕，吃内存
   
   with open('A.csv','rb',encoding='utf-8') as f:
	reader=csv.reader(f)
        headers=next(reader)#先把头读出来No.,Name,Age,Score
	for i,row in enumerate(reader):
	    print("第%d行数据为:"%(i+1),row,"\n")
	    if row["Name"]=="Ben":
		print(row)

  with open("B.csv","wb") as f:
	f_write=csv.writer(f)
	f_write.writerow(['No.','Name','Age','Score'])
	f_write.writerows([(1,'Apple',12,98),(2,'Ben',13,97)],...)
2.字典模式
  with open('A.csv','rb',encoding='utf-8') as f:
	f_csv=csv.DictReader(f)
	for row in f_csv:
	    print(row)#{'Age': '12', 'No.': '1', 'Score': '98', 'Name': 'Apple'}

  with open("B.csv","wb") as f:
	f_csv=csv.DictWriter(f,['No.','Name','Age','Score'])
	f_csv.writeheader()
	f_csv.writerow({'No.':1,'Name':'Apple','Age':12,'Score':98})
	...
     		     
       



#######################################
1. 对应随机打乱
    x=[1,2,3,4,5]
    y=[2,3,4,5,6]
    np.random.seed(113)
    np.random.shuffle(x)#[2, 1, 4, 3, 5]
    np.random.seed(113)
    np.random.shuffle(x)# [3, 2, 5, 4, 6]    
    x_train = np.array(x[:int(len(x) * (1 - 0.2))])#取80%的数据做训练数据，20%做测试数据
    y_train = np.array(y[:int(len(x) * (1 - 0.2))])
    x_test = np.array(x[int(len(x) * (1 - 0.2)):])
    y_test = np.array(y[int(len(x) * (1 - 0.2)):])
    
    mini_batch=50
    for epoch in range(epochs):
	train_date=[x_train,y_train]
        random.shuffle(train_date)
        mini_batches=[train_data[k:k+minibatch] for k in range(0,len(train_date),mini_batch)]
        for mini_batche in mini_batches:
	    batch_x=mini_batch[:,0]
            batch_y=mini_batch[:,1]
        #or
	i=0
        while i< len(train_date):
	    start=i
	    end=i+batch_size
	    batch_x=train_date[:,0][start:end]
	    batch_y=train_date[:,1][start:end] 
	    .......
	    i+=batch_size
        
        



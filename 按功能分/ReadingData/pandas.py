import pandas as pd 
from pandas import Series, DataFrame

1. Series 是一维array-like
    obj = Series([4, 7, -5, 3])
	'''
	0    4
	1    7
	2   -5
	3    3
	dtype: int64
	'''
    obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
    obj2.values# array([ 4,  7, -5,  3])类型为array
    obj2.index# Index(['d', 'b', 'a', 'c'], dtype='object')
    list(obj2.index)
    obj2['a']#-5
    obj2[['d','b']]#obj2[[0,1]]
    	'''
	d   4
	b   7
	dtype: int64
       '''
    obj2['a']=1#修改值
    obj2[obj2>0]        	obj2*2		np.exp(obj2)#都是对其values的运算
	'''
	d    4
	b    7
	c    3
	dtype: int64
	'''
    'b' in obj2 #True
    #字典初始化Series
    sdata = {'Ohio': 35000, 'Texas': 71000}
    obj3 = Series(sdata)
       '''
	Ohio      35000
	Oregon    16000
	dtype: int64
	'''
    #判断是否为NaN数据
    pd.isnull(obj3)
       '''
	Ohio      False
	Oregon    False
	'''
    #重新定义reindex赋值
    obj2=obj.reindex(['a', 'b', 'c', 'd', 'e'])
    #删除
    obj3=obj2.drop('c')
2. DataFrame     表
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
	'year': [2000, 2001, 2002, 2001, 2002]}
    frame = DataFrame(data)
	'''
	    state  year
	0    Ohio  2000
	1    Ohio  2001
	2    Ohio  2002
	3  Nevada  2001
	4  Nevada  2002
	'''
    
    DataFrame(data, columns=['year', 'state'])#按这个列顺序排列先year 后state    
    DataFrame(data, columns=['year', 'state'],index=['one', 'two', 'three', 'four', 'five'])#设定index
    frame.columns#Index(['state', 'year'], dtype='object')
    frame.index#Index(['one', 'two', 'three', 'four', 'five'], dtype='object')
    frame.year#frame['year']  frame[[0]]              frame[['year','state']]==frame[[0,1]]
	'''
	one      2000
	two      2001
	three    2002
	four     2001
	five     2002
	Name: year, dtype: int64

	'''
    frame.ix['one']#frame.ix[0] 取行,变成Series类型
	'''
	year     2000
	state    Ohio
	pop       NaN
	debt      NaN
	Name: one, dtype: object
	'''
    frame.ix['one']['year']#frame.ix['one',['year']]frame.ix[0,0]==frame.ix[0][0]==frme['year']['one']#2000
    frame['year']=2000#将这一列全改为2000
    frame['year']=np.arange(5.)#将这一列改为0 1 2 3 4
    frame.dtypes 
    #使用Series修改
    val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
    frame['debt'] = val
    #
    pop = {'Nevada': {2001: 2.4, 2002: 2.9},'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    frame3 = DataFrame(pop)
	'''
	      Nevada  Ohio
	2000     NaN   1.5
	2001     2.4   1.7
	2002     2.9   3.6
	'''
    #行列对换
    frame3.T
    #values 为array类型
    frame3.values
	'''
array([[ nan,  1.5],
       [ 2.4,  1.7],
       [ 2.9,  3.6]])
	'''
    #删除drop
    frame2.drop('one')#默认是删除行   frame2.drop(['one','two'])
    frame2.drop('year',axis=1)#删除year列  
    #比较
    frame3<2
	'''
	     Nevada   Ohio
	2000  False   True
	2001  False   True
	2002  False  False
	'''
    frame3[frame3<2]=0
    	'''
	      Nevada  Ohio
	2000     NaN   0.0
	2001     2.4   0.0
	2002     2.9   3.6

	'''
     #排序
     frame.sort_index()#按index名小大排序
     frame.sort_index(axis=1)#按columns名小大排序
     frame.sort_index(by='year')#按year列的值小大排序
     fram.sort_index(by=['year','state'])#先按year排序，如果存在相同的值则相同的部分按state排序
     #求和
     frame.sum()#默认是对columns求和
	'''
	pop                          12.1
	state    OhioOhioOhioNevadaNevada
	year                        10006
	dtype: object
	'''
     frame.sum(axis=1，skipna=False)#对行求和,不跳过NaN数据
	'''
	0    2001.5
	1    2002.7
	2    2005.6
	3    2003.4
	4    2004.9
	dtype: float64
	'''
     frame['pop'].sum()#对pop列求和
     #求均值
     frame.mean(axis=1,skipna=False)   
     #求最大值的index系数
     frame['year'].idxmax()#2
     #去除NaN数据
     frame.dropna()#frame[frame.nonull()]去除存在NaN的行
     frame.dropna(how='all')#去除整行或列为NaN的行或列
     #填充
     frame.fillna(0)#将所有NaN的数据填充为0
     frame.fillna({'year':222,'pop':2})#将列year pop中NAN数据分别填充为222 2
     frame.fillna(method='ffill',limit=3)#按列填充最多填充3个数据，由上一个非NAN数据填充
     fram.fillna(data.mean(skipna=True))#均值填充
3. Reading and Writing Data in Text Format  CSV
   #read_csv Load delimited data from a file, URL, or file-like object. Use comma as default delimiter  
	'''
	   a   b   c   d message
	   1   2   3   4   hello   
	   5   6   7   8   world
	   9  10  11  12    foll
	'''
    df=pd.read_csv('ex1.csv',sep=',')#如果不需要第一行做为head,可设header=None,第一行表头默认为0 1 2 3 4，实际数据为数据
	'''
	   a   b   c   d message
	0  1   2   3   4   hello
	1  5   6   7   8   world
	2  9  10  11  12    foll
	'''
    df=pd.read_csv('ex1.csv',sep=',',names=['a', 'b', 'c', 'd', 'e'])#手动另外添加头
   	'''
   	   a   b   c   d        e
	0  a   b   c   d  message
	1  1   2   3   4    hello
	2  5   6   7   8    world
	3  9  10  11  12     foll
'''
     names = ['a', 'b', 'c', 'd', 'message']
     pd.read_csv('ex1.csv', names=names, index_col='message')
	''''
	         a   b   c   d
	message               
	message  a   b   c   d
	hello    1   2   3   4
	world    5   6   7   8
	foll     9  10  11  12
    	'''
    pd.read_csv('ex1.csv', index_col='message')#设定nrows=2，只读取2行数据，foll那行不读取
	''''
	         a   b   c   d
	message               
	hello    1   2   3   4
	world    5   6   7   8
	foll     9  10  11  12
	'''
    #
	'''
	# hey!
	a,b,c,d,message
	# just wanted to make things more difficult for you
	# who reads CSV files with computers, anyway?
	1,2,3,4,hello
	5,6,7,8,world
	9,10,11,12,foo
	'''
    pd.read_csv('ex3.csv',skiprows=[0, 2, 3])#跳过0 ，2，3 行
	'''
	   a   b   c   d message
	0  1   2   3   4   hello
	1  5   6   7   8   world
	2  9  10  11  12     foo
	'''

  


  (2)写
     a=pd.read_csv('ex1.csv', index_col='message')
     a.to_csv('ex1_w.csv')#a.to_csv(sys.stdout, sep=',')可使他print出来
     '''
	message,a,b,c,d
	hello,1,2,3,4
	world,5,6,7,8
	foll,9,10,11,12
	'''

  (3). import csv
     f=open('ex1.csv')
     reader=csv.reader(f)
     for line in reader:
	print(line)
	'''
	['a', 'b', 'c', 'd', 'message']
	['1', '2', '3', '4', 'hello']
	['5', '6', '7', '8', 'world']
	['9', '10', '11', '12', 'foll']
	'''#里面数据全变成字符串类型
     lines=list(csv.reader(f))
     header,values=lines[0],lines[1:]
     data_dict={h:v for h,v in zip(header,zip(*values))}
     '''
	{'a': ('1', '5', '9'),
	 'b': ('2', '6', '10'),
	 'c': ('3', '7', '11'),
	 'd': ('4', '8', '12'),
	 'message': ('hello', 'world', 'foll')}
	'''
      #
      header=next(reader)
      convert=[int,int,int,int,str]
      for line in reader:
          row=list(conv(value) for conv,value in zip(convert,line))
	  print row
	'''
	['a', 'b', 'c', 'd', 'message']	
	[1, 2, 3, 4, 'hello']
	[5, 6, 7, 8, 'world']
	[9, 10, 11, 12, 'foll']
	'''#将数字部分转换
      
   (4). group
	df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
         'key2' : ['one', 'two', 'one', 'two', 'one'],
         'data1' : np.random.randn(5),
         'data2' : np.random.randn(5)})
	'''
	      data1     data2 key1 key2
	0  0.857206  0.065442    a  one
	1  0.691050  0.379452    a  two
	2 -0.174299  0.627501    b  one
	3  0.160182  1.871098    b  two
	4 -0.758572 -0.098099    a  one
	'''
        grouped=df['data1'].groupby(df['key1'])#按照key1对data1进行分组 #df.groupby('key1')['data1']
	list(grouped)[0]
	'''
	('a', 
	 0    0.857206
	 1    0.691050
	 4   -0.758572
	 Name: data1, dtype: float64)
	'''
         gd=grouped.mean()
	'''
	key1
	a    0.263228
	b   -0.007059
	Name: data1, dtype: float64
 	'''
	gd['a']#0.263228
        #
	a=df['data1'].groupby([df['key1'], df['key2']])
	list(a)
	'''
	[(('a', 'one'), 0    0.857206
                        4   -0.758572  Name: data1, dtype: float64), 
	 (('a', 'two'), 1    0.69105   Name: data1, dtype: float64), 
	 (('b', 'one'), 2   -0.174299  Name: data1, dtype: float64), 
	 (('b', 'two'), 3    0.160182  Name: data1, dtype: float64)]
	'''
	b=a.mean()
	'''
	key1  key2
	a     one     0.049317
	      two     0.691050
	b     one    -0.174299
	      two     0.160182
	Name: data1, dtype: float64
	'''
	b['a']['one']#0.049317
	b.unstack()#转化为标准型
	'''
	key2       one       two
	key1                    
	a     0.049317  0.691050
	b    -0.174299  0.160182
	'''
	df.groupby('key1').mean()
	'''
	         data1     data2
	key1                    
	a     0.263228  0.115599
	b    -0.007059  1.249299
	'''#因为key2不是数字格式，默认为不入计算结果
	df.groupby(['key1', 'key2']).mean()
	'''
	              data1     data2
	key1 key2                    
	a    one   0.049317 -0.016328
	     two   0.691050  0.379452
	b    one  -0.174299  0.627501
	     two   0.160182  1.871098
	'''
	


4. TXT
	'''
	        A        B        C 
	aaa -0.264438 -1.026059 -0.619500
	bbb 0.927272 0.302904 -0.032399
	ccc -0.264273 -0.386314 -0.217601
	ddd -0.871858 -0.348382 1.100491
	'''
    a=open('ex2.txt')
    list(a)
	'''
	['        A        B        C \n',
	 'aaa -0.264438 -1.026059 -0.619500\n',
	 'bbb 0.927272 0.302904 -0.032399\n',
	 'ccc -0.264273 -0.386314 -0.217601\n',
	 'ddd -0.871858 -0.348382 1.100491\n']
	'''
     b=pd.read_table('ex2.txt',sep='\s+')
	'''
	            A         B         C
	aaa -0.264438 -1.026059 -0.619500
	bbb  0.927272  0.302904 -0.032399
	ccc -0.264273 -0.386314 -0.217601
	ddd -0.871858 -0.348382  1.100491
	'''
5. 读取json数据

    import json

    obj = """
    {
    "siblings": {"name": "Scott", "age": 25, "pet": "Zuko"},
    "mary":{"name": "Katie", "age": 33, "pet": "Cisco"}
    }
    """
    result = json.loads(obj)
    cc=pd.DataFrame(result)
    '''
           mary siblings
    age      33       25
    name  Katie    Scott
    pet   Cisco     Zuko
   '''




'''
# Linear Regression: Decomposition Method
#----------------------------------
#
# Given Ax=b, solving for A:  y=a1*x1+a2*x2, a1=1
  for the sample  number of m 
  XA=Y     shapel X:[m 2]  A:[2 1] Y:[m 1]
# This function shows how to use Tensorflow to
# solve linear regression via the matrix inverse.
why:comuter the INVERSE  (X.T*X)^(-1)  is inefficient when m ,n is big 
# Given Ax=b, and a Cholesky decomposition such that
#  X.T*X= L*L' then we can get solve for A via
 X*A=Y->X.T*X*A=X.T*Y  s.t:X.T*X=L*L' ->L*L'*A=X.T*Y
 s.t:L'*A=Z  -> L*Z=X.T*Y
# 1) L*Z=X.T*Y  -> Z
# 2) L'*A=Z     -> A
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
x_vals=np.linspace(0,10,100)
y_vals=x_vals+np.random.normal(0,1,100)
x_vals_column=np.transpose(np.matrix(x_vals))
ones_column=np.matrix(np.ones((100,1)))#ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
x=np.column_stack((ones_column,x_vals_column))
y=np.matrix(y_vals).T#y=np.transpose(np.matrix(y_vals))

#create tensors
x_tensor=tf.constant(x)
y_tensor=tf.constant(y)
# Find Cholesky Decomposition
x_xt=tf.matmul(tf.transpose(x_tensor),x_tensor)
L=tf.cholesky(x_xt)
#solve L*Z=X.T*Y 
xt_y=tf.matmul(tf.transpose(x_tensor),y_tensor)
Z=tf.matrix_solve(L,xt_y)
#solve L'*A=Z 
A=tf.matrix_solve(tf.transpose(L),Z)
A_eval=sess.run(A)
print("A:\n",A_eval)

#get the best fit line
best_fit_y=[]
for i in x_vals:
    best_fit_y.append(A_eval[0]+i*A_eval[1])
#best_fit_y=(np.dot(x,np.matrix(A_eval))).tolist()


#plot the fit line 
plt.plot(x_vals,y_vals,"o",label="Data")
plt.plot(x_vals,best_fit_y,'r-',label="Best fit line",linewidth=3)
plt.legend(loc='upper letf')
plt.show()
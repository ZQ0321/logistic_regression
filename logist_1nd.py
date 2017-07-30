# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 21:04:04 2017

@author: Administrator
"""
from __future__ import absolute_import


from import_data_a import read_data_sets
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def model(x,w):
    #return 1.0/(1.0+tf.exp(-(tf.matmul(x,w))))
    return tf.matmul(x,w)
    
learning_rate=0.01
num_epoch=10000  


#调用会话
#sess=tf.InteractiveSession()

#导入数据
data_sets,odata=read_data_sets()
#为输入和输出创建占位符
x=tf.placeholder(dtype="float",shape=[None,944])
y_=tf.placeholder(dtype="float",shape=[None,1])

#创建变量
w=tf.Variable(tf.random_normal([944,1],stddev=0.01,name='weight'))  
#b=tf.Variable(tf.random_normal([1],name='bias')) 
y_predict=model(x,w)
#num_samples=data_sets.train.images.shape[0]
num_samples=data_sets.train._num_examples
#cost=tf.reduce_sum(y_*tf.log(y_predict)+(1-y_)*tf.log(1-y_predict))
cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_predict,y_))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
predict_op=tf.nn.sigmoid(y_predict)
  
with tf.Session() as sess:  
    #初始化所有变量  
    sess.run(tf.initialize_all_variables())  
    #开始训练  
    for epoch in range(num_epoch):
        batch1=data_sets.train.next_batch(50)
        sess.run(optimizer,feed_dict={x:batch1[0],y_:batch1[1]})
    predicts,cost_=sess.run([predict_op,cost],feed_dict={x:data_sets.test.images\
                            ,y_:data_sets.test.labels})
    print ('final auc:',roc_auc_score(data_sets.test.labels,predicts),cost_)
    
odata.close()


"""
得到的final auc:0.999,cost_:0.0311
"""
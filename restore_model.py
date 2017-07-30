# -*- coding: utf-8 -*-
"""
The practice of importing models for new train_datas.
"""


import tensorflow as tf

#导出模型和变量
sess=tf.Session()
saver=tf.train.import_meta_graph(r'D:\CBP\model.ckpt-999.meta')
saver.restore(sess,tf.train.latest_checkpoint(r'D:\CBP'))

#输出保存的模型的参数
print (sess.run(['hidden/weight:0','hidden/biases:0','logits/weight:0']))


#得到placeholder_tensor，用新数据填占位符来训练模型
graph=tf.get_default_graph()
images=graph.get_tensor_by_name("Placeholder:0")
labels=graph.get_tensor_by_name("Placeholder_1:0")

"""
填入新数据
feed_dict={images:,labels:,}
"""

#查看保存模型的损失函数值
original_loss_value=graph.get_collection('loss')

#得到predicts,loss节点和训练操作
predicts=graph.get_tensor_by_name("logits/Sigmoid:0")
loss=graph.get_tensor_by_name("Mean:0")
train_op=graph.get_operation_by_name('GradientDescent')

"""下面可以用新的数据训练模型"""

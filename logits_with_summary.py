# -*- coding: utf-8 -*-

"""A logistic regression for DATAS transformed form CBP-4trace.

Accuracy:
With fixed learning_rate, this file achieves ~99.1635% accuracy after 10k steps

Applying exponential_decay to learning_rate, it achieves ~ 99.8332% accuracy
after 10k steps.

After applying moving average to trainable_variables, it achieves ~99.9874%
accuracy after 10k steps.(Here,using this skill just to practice. The improvement
will be remarkable in deep networks.)
"""

from import_data_new import read_data_sets
import tensorflow as tf
from six.moves import xrange
from sklearn.metrics import roc_auc_score
import os


FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size',50,\
'Batch size. Must divide evenly into the dataset sizes.')

tf.app.flags.DEFINE_integer('max_steps',10000,'Number of steps to run trainer')

tf.app.flags.DEFINE_string('log_dir',r'D:\CBP','the directory to put log data')

#for fixed learning_rate
#tf.app.flags.DEFINE_float('learning_rate',0.01,'Initial learning rate.')


#定义全局变量
initial_learning_rate=0.1
decay_steps=3000
decay_rate=0.1
moving_average_decay=0.9999

"""定义函数计算模型输出"""
def inference(images):
    #定义参数
    W=tf.Variable(tf.random_normal([944,1],stddev=0.01))

    #计算模型输出
    logits=tf.matmul(images,W)
    labels_=tf.sigmoid(logits)
    tf.histogram_summary('The_output_labels',labels_)
    return logits,labels_

"""计算模型损失"""
def loss(logits,labels):
    #计算模型损失
    cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(logits,labels)
    loss_value=tf.reduce_mean(cross_entropy)
    tf.add_to_collection('loss',loss_value)
    return loss_value
    
"""定义训练函数"""
def train(loss,global_step):
    """
    常规得到训练节点
    #训练节点
    lr=tf.train.exponential_decay(0.1,global_step,3000,0.1,staircase=True)
    train_op=tf.train.GradientDescentOptimizer(lr).minimize(loss,\
global_step=global_step)
    """
    
    """对参数应用指数衰减"""
    #指数衰减学习率
    lr=tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,\
                                  decay_rate,staircase=True)
    tf.scalar_summary('learning_rate',lr)
    
    #计算和应用梯度
    opt=tf.train.GradientDescentOptimizer(lr)
    gradients=opt.compute_gradients(loss)
    apply_gradients_op=opt.apply_gradients(gradients)
    
    #对参数用movingaverege
    variable_average=tf.train.ExponentialMovingAverage(moving_average_decay,\
                                                       global_step)
    variable_average_op=variable_average.apply(tf.trainable_variables())
    
    #用控制环境得到训练节点
    with tf.control_dependencies([apply_gradients_op,variable_average_op]):
        train_op=tf.no_op(name='train')
    
    return train_op

"""运行训练函数"""
def run_training():
    #读取数据
    data_sets,odata=read_data_sets()
    with tf.Graph().as_default():
        
        global_step=tf.Variable(0.0,trainable=False)
        
        images=tf.placeholder(dtype="float",shape=[None,944])
        labels=tf.placeholder(dtype="float",shape=[None,1])
        logits,labels_=inference(images)
        loss_value=loss(logits,labels)
        
        train_op=train(loss_value,global_step) 
        init_op=tf.global_variables_initializer()
        
        summary=tf.merge_all_summaries()
        saver=tf.train.Saver()
        #定义会话
        with tf.Session() as sess:
            summary_writter=tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
            sess.run(init_op)
            #开始训练
            for step in xrange(FLAGS.max_steps):
                batch=data_sets.train.next_batch(FLAGS.batch_size)
                feed_dict={images:batch[0],labels:batch[1]}
                sess.run([train_op],feed_dict=feed_dict)
                
                #每一百步写入一次summary
                if step % 100==0:
                    summary_str=sess.run(summary,feed_dict=feed_dict)
                    summary_writter.add_summary(summary_str,global_step=step)
                    summary_writter.flush()
                
                #每999步保存一次变量
                if (step+1) % 1000==0 or (step+1)==FLAGS.max_steps:
                    path=os.path.join(FLAGS.log_dir,'model.ckpt')
                    saver.save(sess,path,global_step=step)
                    
                    
                
            #计算模型在测试样本上的准确率
            feed_dict={images:data_sets.test.images,labels:data_sets.test.labels}
            labels_=sess.run(labels_,feed_dict=feed_dict)
            precision=roc_auc_score(data_sets.test.labels,labels_)
            print ('final auc: %f' %(precision))
            
    odata.close()
            
"""定义主函数"""
def main(argv='None'):
    """
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
    """ 
    run_training()
 
    
if __name__=='__main__':
    tf.app.run()
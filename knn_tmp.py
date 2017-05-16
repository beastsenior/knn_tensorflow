#成功改版为k=N的，可自己设置，但好像k=1还是效果最好

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)

K=7  #K临近的K值

#导入mnist数据
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_data/",one_hot=True)

#计算图
xtrain=tf.placeholder("float",[None,784])
xtest=tf.placeholder("float",[784])

distance=tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))),axis=1)
pred=tf.arg_min(distance,0)

init=tf.global_variables_initializer()

#启动会话
with tf.Session() as sess:
    sess.run(init)
    accuracy = 0.
    Xtest, Ytest = mnist.test.next_batch(1000) #测试数据集和标签
    Ntest=len(Xtest)
    for i in range(Ntest):
        pred_class_label = [0] * 10
        for j in range(K):
            # 训练数据集和标签
            Xtrain, Ytrain = mnist.train.next_batch(40000)
            nn_index = sess.run(pred, feed_dict={xtrain: Xtrain, xtest: Xtest[i]})
            pred_class_label[np.argmax(Ytrain[nn_index])] +=1
        pred_class_label_all=np.argmax(pred_class_label)
        true_class_label=np.argmax(Ytest[i])
        if pred_class_label_all==true_class_label:
            accuracy+=1
        else:
            print('Step:',i) #第i步
            print('Wrong:',true_class_label, '=>', pred_class_label_all) #真实（正确）=>预测
            print('Vote Table:', pred_class_label)  #具体投票情况
    print("done!!!")
    accuracy /=Ntest
    print("accuracy:", accuracy)  #正确率



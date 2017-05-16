#实验用改版，但基本和原版相同，k=1

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)

#导入mnist数据
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_data/",one_hot=True)

#数据集和标签
Xtrain,Ytrain=mnist.train.next_batch(5000)
Xtest, Ytest= mnist.test.next_batch(100)
print("Xtrain.shape: ", Xtrain.shape, "Xtest.shape: ",Xtest.shape)
print("Ytrain.shape: ", Ytrain.shape, "Ytest.shape: ",Ytest.shape)

#计算图
xtrain=tf.placeholder("float",[None,784])
xtest=tf.placeholder("float",[784])

distance=tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))),axis=1)
nearest_node=tf.arg_min(distance,0)
tmp=Ytrain[nearest_node]
nearest_node_label = tf.arg_max(tmp,0)

init=tf.global_variables_initializer()

#启动会话
with tf.Session() as sess:
    sess.run(init)
    accuracy = 0.
    Ntest=len(Xtest)
    for i in range(Ntest):
        pred_class_label = sess.run(nearest_node_label, feed_dict={xtrain: Xtrain, xtest: Xtest[i, :]})
        true_class_label=np.argmax(Ytest[i])
        print("test",i,"predicted class label:",pred_class_label,"true class label:",true_class_label)
        if pred_class_label==true_class_label:
            accuracy+=1
    print("done!!!")
    accuracy /=Ntest
    print("accuracy:", accuracy)

    print('nn_index:', nn_index)
    print('nn_index type:',type(nn_index))
    print('Ytest[nn_index]:',Ytrain[nn_index])
    print('pred_class_label:', pred_class_label)
    print('pred_class_label type:',type(pred_class_label))
    print('true_class_label:', true_class_label)
    print('true_class_label type:',type(true_class_label))

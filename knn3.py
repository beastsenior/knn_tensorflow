#失败版本，想直接输出标签，但是失败了，主要在tmp位置

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
Xtest, Ytest= mnist.test.next_batch(1000)
print("Xtrain.shape: ", Xtrain.shape, "Xtest.shape: ",Xtest.shape)
print("Ytrain.shape: ", Ytrain.shape, "Ytest.shape: ",Ytest.shape)

#计算图
xtrain=tf.placeholder("float",[None,784])
xtest=tf.placeholder("float",[784])

tmp=tf.add(xtrain, tf.negative(xtest))
distance=tf.reduce_sum(tf.abs(tmp),axis=1)

pred=tf.arg_min(distance,0)

init=tf.global_variables_initializer()

#启动会话
with tf.Session() as sess:
    sess.run(init)
    accuracy = 0.
    Ntest=len(Xtest)
    for i in range(Ntest):
        nn_index=sess.run(pred,feed_dict={xtrain:Xtrain,xtest: Xtest[i,:]})
        pred_class_label=np.argmax(Ytrain[nn_index])
        true_class_label=np.argmax(Ytest[i])
        print("test",i,"predicted class label:",pred_class_label,"true class label:",true_class_label)
        if pred_class_label==true_class_label:
            accuracy+=1
    print("done!!!")
    accuracy /=Ntest
    print("accuracy:", accuracy)

    # print('pred:', pred)
    # print('pred type:',type(pred))
    #
    # print('nn_index:', nn_index)
    # print('nn_index type:',type(nn_index))
    #
    # print('Ytest[nn_index]:',Ytrain[nn_index])
    # print('Ytest[nn_index] type:',type(Ytrain[nn_index]))
    #
    # print('pred_class_label:', pred_class_label)
    # print('pred_class_label type:',type(pred_class_label))
    #
    # print('true_class_label:', true_class_label)
    # print('true_class_label type:',type(true_class_label))

import tensorflow as tf
import os
import numpy as np
import pickle

feature_num=12
TIME_STEP=240

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

test_data = pickle.load(open("test_dataset.pkl", 'rb'))
train_data = pickle.load(open("train_dataset.pkl", 'rb'))

Xtrain=train_data.dataset.features
Xtest=test_data.dataset.features
Ytrain=train_data.dataset.labels
Ytest=test_data.dataset.labels

print('Xtrain.shape:',Xtrain.shape,'Xtest.shape:',Xtest.shape)
print('Ytrain.shape:',Ytrain.shape,'Ytest.shape:',Ytest.shape)
	
xtrain=tf.placeholder('float',[None,TIME_STEP,feature_num],name='X_train')   #不知道训练样本的个数用None来表示，特证数是180
xtest=tf.placeholder('float',[TIME_STEP,feature_num],name='X_test')
	
distance=tf.reduce_sum(tf.abs(tf.add(xtrain,tf.negative(xtest))),axis=1)  #逐行进行缩减运算，最终得到一个行向量
	
pred=tf.arg_min(distance,0)   #获取最小距离的索引
	
init=tf.global_variables_initializer()

TP_list=[]
FP_list=[]
TN_list=[]
FN_list=[]
acc_list=[]
pre_list=[]
rec_list=[]
F1_list=[]

for run in range(10):
    train_data.dataset.shuffle()
    test_data.dataset.shuffle()

    Xtrain=train_data.dataset.features
    Xtest=test_data.dataset.features
    Ytrain=train_data.dataset.labels
    Ytest=test_data.dataset.labels
    #TP (a fall occurs; the algorithm detects it), 
    #TN (a fall does not occur; the algorithm doesnot detect a fall), 
    #FP (a fall does not occur but the algorithm reports a fall),
    #FN (a fall occursbut the algorithm misses it)
    TP=0
    FP=0
    TN=0
    FN=0

    with tf.Session() as sess:
        sess.run(init)
        Ntest=len(Xtest)  #测试样本数量
        for i in range(Ntest):
            nn_index=sess.run(pred,feed_dict={xtrain:Xtrain,xtest:Xtest[i,:]})   #每次只传入一个测试样本
            pred_class_label=np.argmax(Ytrain[nn_index])
            true_class_label=np.argmax(Ytest[i])
            #print('Test',i,'Prediction Class Label:',pred_class_label,'True Class Label:',true_class_label)

            # Fall:01 ADL:10
            if (true_class_label == 0):
                if pred_class_label==true_class_label:
                    TP+=1
                else:
                    FN+=1
            else:
                if pred_class_label==true_class_label:
                    TN+=1
                else:
                    FP+=1

        print('Done!')
        accuracy=(TP + TN)/Ntest
        precision=TP/(TP+FP)
        recall = TP / ( TP + FN )
        F1 = ( 2* precision * recall ) / (precision + recall)

        TP_list.append(TP)
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        acc_list.append(accuracy* 100)
        pre_list.append(precision* 100)
        rec_list.append(recall* 100)
        F1_list.append(F1* 100)

        print('Accuracy:',accuracy * 100)
        print('precision:',precision* 100)
        print('recall:',recall* 100)
        print('F1:',F1* 100)

print("TP_list:",TP_list)
print("TN_list:",TN_list)
print("FP_list:",FP_list)
print("FN_list:",FN_list)
print("acc_list:",acc_list)
print("pre_list:",pre_list)
print("rec_list:",rec_list)
print("F1_list:",F1_list)

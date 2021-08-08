import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, GRU, Dropout
from tensorflow.keras.models import Sequential
import os
import json
import pickle
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import time
import noise
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import math


#一些基本设定
checkpoint_flag = int(input("是否进行断点续训？是：1；否：0")) #询问是否使用以往最优的参数进行测试
snr=0  #目标领域添加高斯白噪音后的信噪比

#样本参数与预测长度选择
train_len=100  #训练集，一个训练（验证）样本的长度
pre_len=50  #将要预测的值的长度(同时也是验证集数据长度)
train_num=3000  #训练集样本个数
test_num=300  #验证集样本数

batch_size=256
epochs=50


data = sio.loadmat(r'data\0HP\normal_0_97.mat')
de_data = data['X097_DE_time']

training_set = de_data[0:train_len + train_num]  #训练集（还未划分训练样本）
test_set = de_data[train_len + train_num:2*train_len + train_num + test_num]  #测试集（还未划分测试样本）

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化

x_train = []
y_train = []

x_test = []
y_test = []

# 训练集：2048样本点+3000样本数=5048个点数据
# 利用for循环，遍历整个训练集，提取训练集中连续2048个点作为输入特征x_train，第2049个数据点作为标签，for循环共构建5048-2048=3000个训练样本
for i in range(train_len, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - train_len:i, 0])
    y_train.append(training_set_scaled[i, 0])

# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
# 使x_train符合GRU输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]即3000组数据；输入2048个点，预测出第2049个点，循环核时间展开步数为2048; 每个时间步送入的特征是某一个采样时刻的值，，只有1个数据，故每个时间步输入特征个数为1
x_train = np.reshape(x_train, (x_train.shape[0], train_len, 1))

# 验证集：300个样本
for i in range(train_len, len(test_set)):
    x_test.append(test_set[i - train_len:i, 0])
    y_test.append(test_set[i, 0])

# 测试集变array并reshape为符合GRU输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], train_len, 1))





model_name = "predict-1-GRU"
# 实例化一个Sequential
model = Sequential()
#第一层GRU
model.add(GRU(80, activation='tanh', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                    bias_initializer='zeros', return_sequences=True))
model.add(Dropout(0.2))
# #第二层GRU
model.add(GRU(100, return_sequences=False))
model.add(Dropout(0.2))
# #第三层GRU
# model.add(GRU(120, return_sequences=False))
# model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值

logdir = os.path.join('.\logs\predict-GRU-1_logs')
summary = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)


now_time = time.time()  #记录训练开始时间
if checkpoint_flag :
    print('开始断点续训')
    checkpoint_save_path = "./checkpoint/predict-1-GRU.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                        save_weights_only=True,
                                                        save_best_only=True)
    # 开始模型训练
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test), shuffle=True,
                        callbacks=[cp_callback,summary])
else :#开始模型训练
    print('未进行断点续训')
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_test, y_test), shuffle=True,
                        callbacks=[summary])

total_time = time.time() - now_time  #记录训练总时间
#打印训练、测试耗时
print("训练总耗时/s：", total_time) #打印训练总耗时

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


################## predict ######################
predicted_data=[]  #保存预测数据
source_3=x_test[0:1]  #把第一个测试集样本抽出来，作预测.三维数组
source_1=x_test[0].ravel()  #第一个测试集样本，一维数组

for i in range(pre_len):
    pre = model.predict(source_3)  #用一个三维样本输入模型
    pre=pre.ravel()
    predicted_data = np.append(predicted_data, pre[0])  #将新预测得到的结果保存进预测数组
    source_1 = np.append(source_1[1:train_len+1], pre[0])  #更新一维样本
    source_3 = np.reshape(source_1, (1, train_len, 1))  #将一维样本升高至三维，以供下次输入模型



predicted_data = np.reshape(predicted_data, (pre_len, 1))
# 对原始数据还原---从（0，1）反归一化到原始范围
source_data = sc.inverse_transform(test_set[0:train_len])
# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_data = sc.inverse_transform(predicted_data)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_data = sc.inverse_transform(test_set[train_len:train_len+pre_len])
# 画出真实数据和预测数据的对比曲线
plt.plot(real_data, color='red', label='real_data')
plt.plot(predicted_data, color='blue', label='Predicted_data')
plt.title('data Prediction')
plt.xlabel('Time')
plt.ylabel('accelarate')
plt.legend()
plt.show()

#保存预测数据与真实数据
np.savetxt(r'save_txt\source_data.txt',source_data)
np.savetxt(r'save_txt\predict_data.txt',predicted_data)
np.savetxt(r'save_txt\real_data.txt',real_data)


##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_data, real_data)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_data, real_data))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_data, real_data)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)



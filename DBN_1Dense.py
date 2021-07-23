import tensorflow as tf
from sklearn import datasets
from sklearn.neural_network import BernoulliRBM
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import preprocess
import numpy as np
import os
from matplotlib import pyplot as plt


# 训练参数
batch_size = 256
epochs = 100
num_classes = 10
length = 4096
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例

path = r'data\0HP'

x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28) #最后俩个：是否数据增强，数据增强的顺延间隔
components = 100
#RBM玻尔兹曼机部分
rbm = BernoulliRBM(n_components=components ,n_iter =40 ,learning_rate =0.01 , verbose = False)
rbm.fit(x_train)

#读取rbm的参数，以供后面提取
rbm_biases=rbm.intercept_hidden_
rbm_weights= rbm.components_

#微调阶段
model_name = "1_DBN_cnn_1D"

# 实例化一个Sequential
model = Sequential()

# 添加全连接层1
model.add(Dense(units=components, activation='relu', kernel_regularizer=l2(1e-4),
                input_dim=4096,name='rbm'))
# 增加输出层，共num_classes个单元
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy',
              metrics=['accuracy'])

#加载rbm参数到第一个全连接层
layers = model.get_layer(name = 'rbm')
layers.set_weights([rbm_weights.transpose(),rbm_biases])

# 开始模型训练
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True)

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])

# 显示训练集和验证集的acc和loss曲线
acc = history.history['accuracy'] #训练集准确率
val_acc = history.history['val_accuracy'] #测试集准确率
loss = history.history['loss'] #训练集损失函数
val_loss = history.history['val_loss'] #测试集损失函数

#将行表转换为列数组，便于保存与处理
s_acc=np.array(acc)
s_acc = s_acc.reshape(s_acc.shape[0],1)
s_val_acc=np.array(val_acc)
s_val_acc = s_val_acc.reshape(s_val_acc.shape[0],1)
s_loss=np.array(loss)
s_loss= s_loss.reshape(s_loss.shape[0],1)
s_val_loss=np.array(val_loss)
s_val_loss= s_val_loss.reshape(s_val_loss.shape[0],1)

# 画出ACC,LOSS曲线
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


#保存acc,loss数据
np.savetxt('./acc.txt',s_acc)
np.savetxt('./val_acc.txt',s_val_acc)
np.savetxt('./loss.txt',s_loss)
np.savetxt('./val_loss.txt',s_val_loss)


# print(model.trainable_variables) #打印训练参数
#将参数读取到
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
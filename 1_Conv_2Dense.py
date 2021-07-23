import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import preprocess
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.utils import plot_model
import time
import noise

snr=0;  #信噪比

checkpoint_flag = int(input("是否进行断点续训？是：1；否：0")) #询问是否使用以往最优的参数进行测试

# 训练参数
batch_size = 256
epochs = 50
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

#插入新维度，方便卷积网络输入
x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]
# 输入数据的维度
input_shape =x_train.shape[1:]

model_name = "1_Conv_2Dense"

# 实例化一个Sequential
model = Sequential()

#第一层卷积
model.add(Conv1D(filters=32, kernel_size=20, strides=8, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, strides=4, padding='valid'))
model.add(Dropout(0.2))


# 从卷积到全连接需要展平
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=500, activation='relu', kernel_regularizer=l2(1e-4)))
# 增加输出层，共num_classes个单元
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))


# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])


now_time = time.time()  #记录训练开始时间
if checkpoint_flag :
    checkpoint_save_path = "./checkpoint/1_Conv_2Dense.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    # 开始模型训练
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
                        callbacks=[cp_callback])
    print('已进入断点续训')
else: #重新训练
    print('未进行断点续训')
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(x_valid, y_valid), shuffle=True)

total_time = time.time() - now_time  #记录训练总时间

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])


#测试当目标领域分布与源分布差异较大时，模型的自适应情况
target_path = r'data\1HP' #使用马力=1的数据集作为目标领域
t_x_train, t_y_train, t_x_valid, t_y_valid, t_x_test, t_y_test = preprocess.prepro(d_path=target_path,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=[0.2,0.3,0.5],
                                                                  enc=True, enc_step=28) #最后俩个：是否数据增强，数据增强的顺延间隔

x_test=noise.wgn(t_x_test,snr)  #给目标领域信号加入高斯白噪音，其中snr表示信噪比
#插入新维度，方便卷积网络输入
t_x_train, t_x_valid, t_x_test = t_x_train[:,:,np.newaxis], t_x_valid[:,:,np.newaxis], t_x_test[:,:,np.newaxis]

#目标领域数据集测试
t_score = model.evaluate(x=t_x_test, y=t_y_test, verbose=0)
print("目标领域的损失率：", t_score[0])
print("目标领域的准确率：", t_score[1])


# ############################################## show#####################################
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

#保存acc,loss数据
np.savetxt('./acc.txt',s_acc)
np.savetxt('./val_acc.txt',s_val_acc)
np.savetxt('./loss.txt',s_loss)
np.savetxt('./val_loss.txt',s_val_loss)

#画出ACC,LOSS曲线
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

print("训练总耗时/s：", total_time) #打印训练总耗时

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

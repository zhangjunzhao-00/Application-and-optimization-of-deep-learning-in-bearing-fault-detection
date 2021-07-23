from sklearn.neural_network import BernoulliRBM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import pickle
import preprocess
from matplotlib import pyplot as plt
import numpy as np
import keras

# 数据加载
num_class = 10
length = 4096
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例

# 加载数据函数
path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28) #最后俩个：是否数据增强，数据增强的顺延间隔

# DBN网络构建类
class DBN():

    def __init__(
            self,
            train_data,  # 训练集
            targets,  # 训练标签
            layers,  # DBN隐藏层-层数
            outputs,  # 输出类别
            rbm_lr,  # rbm学习率
            rbm_iters,  # rbm学习迭代次数
            rbm_dir=None,
            test_data=None,  # 测试集
            test_targets=None,  # 测试标签
            epochs=10,  # 循环次数
            fine_tune_batch_size=256,  # 一次投喂样本数
    ):

        self.hidden_sizes = layers  # 隐藏层-数量，如【2 3 5 8】
        self.outputs = outputs  # 输出标签有几个类别
        self.targets = targets  # 训练-标签
        self.data = train_data  # 训练集

        if test_data is None:
            self.validate = False  # 如果无测试集，则使用训练集中数据
        else:
            self.validate = True  # 如果有测试集，则不使用训练集中数据

        self.valid_data = test_data  # 测试集
        self.valid_labels = test_targets  # 测试-标签

        self.rbm_learning_rate = rbm_lr  # rbm学习率
        self.rbm_iters = rbm_iters  # rbm循环次数

        self.epochs = epochs  # BP网络循环次数
        self.nn_batch_size = fine_tune_batch_size  # 一次投喂数量

        self.rbm_weights = []  # rbm的权重w，初始为空
        self.rbm_biases = []  # rbm的偏执项b，初始为空
        self.rbm_h_act = []  # rbm隐藏层-输出，初始为空

        self.model = None
        self.history = None



    def pretrain(self, save=True):  # 预训练-函数

        visual_layer = self.data  # 可见层-导入训练集数据

        for i in range(len(self.hidden_sizes)):  # 对每个隐藏层
            print("[DBN] Layer {} Pre-Training".format(i + 1))  # 打印：目前进行到第几层RBM训练了

            # rbm：第i层隐藏层的神经元个数，训练次数，第i层的学习率，True表示输出日志文件，一次投喂20
            rbm = BernoulliRBM(n_components=self.hidden_sizes[i], n_iter=self.rbm_iters[i],
                               learning_rate=self.rbm_learning_rate[i], verbose=True, batch_size=20)
            rbm.fit(visual_layer)  # 训练输入的训练集
            self.rbm_weights.append(rbm.components_)  # 往rbm的权重中，增添新的权重
            self.rbm_biases.append(rbm.intercept_hidden_)  # 往rbm的偏置项中，增添新的偏置项
            self.rbm_h_act.append(rbm.transform(visual_layer))  # 隐层输出，存入h_act

            visual_layer = self.rbm_h_act[-1]  # 使前一层的隐层，成为新rbm的显层


    #开始微调
    def finetune(self):  # 微调过程，输入为self
        model = Sequential()  # 进入全连接网络环节
        for i in range(len(self.hidden_sizes)):  # 对每一个rbm的隐藏层，第i层执行如下操作(替换为全连接神经网络，第一层需设置输入维度dim)

            if i == 0:  # 如果是第一个全连接网络，需要设置输入维度
                model.add(Dense(self.hidden_sizes[i], activation='relu', input_dim=self.data.shape[1],
                                name='rbm_{}'.format(i)))  # 第一层的话规定下输入数据的维度
            else:  # 随后几层无需设置输入维度dim
                model.add(Dense(self.hidden_sizes[i], activation='relu', name='rbm_{}'.format(i)))

        model.add(Dense(self.outputs, activation='softmax'))  # 有n个输出（outputs），softmax
        # 反向传播参数设置
        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])  # 反向传播，优化器，损失函数，判别标准

        for i in range(len(self.hidden_sizes)):  # 每一个rbm的隐藏层，对第i个进行如下操作——将之前rbm预训练得到的参数给到新的bp网络参数
            layer = model.get_layer('rbm_{}'.format(i))  # i=0的rbm名字为'rbm_{0}'
            layer.set_weights([self.rbm_weights[i].transpose(), self.rbm_biases[i]])  # 初始化参数


        if self.validate:  # 如果有测试集，则值为1：不适用训练集中数据；否则为0：使用训练集中数据
            # 训练集，循环次数，一次投喂样本数，使用训练集中数据
            self.history = model.fit(trainx, trainy,
                                     epochs=self.epochs,
                                     batch_size=self.nn_batch_size,
                                     validation_data=(self.valid_data, self.valid_labels))
                                     # callbacks=[checkpointer, tensorboard])
        else:  # 没有测试集，则使用训练集中数据
            self.history = model.fit(trainx, trainy,
                                     epochs=self.epochs,
                                     batch_size=self.nn_batch_size,)
                                     # callbacks=[checkpointer, tensorboard])
        self.model = model  # 将搭建模型幅值给self.model

    def plot_loss_acc (self):
        # 显示训练集和验证集的acc和loss曲线
        acc = self.history.history['accuracy']  # 训练集准确率
        val_acc = self.history.history['val_accuracy']  # 测试集准确率
        loss = self.history.history['loss']  # 训练集损失函数
        val_loss = self.history.history['val_loss']  # 测试集损失函数

        # 将行表转换为列数组，便于保存与处理
        s_acc = np.array(acc)
        s_acc = s_acc.reshape(s_acc.shape[0], 1)
        s_val_acc = np.array(val_acc)
        s_val_acc = s_val_acc.reshape(s_val_acc.shape[0], 1)
        s_loss = np.array(loss)
        s_loss = s_loss.reshape(s_loss.shape[0], 1)
        s_val_loss = np.array(val_loss)
        s_val_loss = s_val_loss.reshape(s_val_loss.shape[0], 1)

        # 保存acc,loss数据
        np.savetxt('./acc.txt', s_acc)
        np.savetxt('./val_acc.txt', s_val_acc)
        np.savetxt('./loss.txt', s_loss)
        np.savetxt('./val_loss.txt', s_val_loss)

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

if __name__ == '__main__':
    trainx = x_train
    trainy = y_train
    testx = x_test
    testy = y_test

    dbn = DBN(train_data=trainx, targets=trainy,          #训练集
              test_data = testx, test_targets = testy,    #测试集
              layers=[100, 50,50],                              #rbm隐藏层
              outputs=num_class,                          #分类数
              rbm_iters=[40, 40,40],                           #每层rbm隐藏层迭代次数
              rbm_lr=[0.01, 0.01,0.01],                        #每层rbm隐藏层学习率
              epochs=100,                                  #微调过程的总迭代次数
              # fine_tune_batch_size=20,                  #一次投喂样本数
              )
    dbn.pretrain(save=True)  #预训练
    dbn.finetune()           #参数微调

    # 评估模型
    score = dbn.model.evaluate(x=x_test, y=y_test, verbose=0)
    print("测试集上的损失率：", score[0])
    print("测试集上的准确率：", score[1])
    dbn.plot_loss_acc()      #画出loss、acc曲线

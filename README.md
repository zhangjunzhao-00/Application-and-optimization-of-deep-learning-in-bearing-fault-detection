# Application and optimization of deep learning in bearing fault detection
**深度学习在轴承故障诊断领域的应用与优化**

**1.简介**

暑期做的小研究，主要解决空载数据集训练，带载数据集测试存在的识别准确率较低的问题

使用CWRU轴承数据库的数据集

**2.硬件**

CPU：i7-8550U，显卡：MX150

**3.框架**

Keras，Sklearn

**4.依赖**

tensorflow 2.0;keras;numpy;scipy;os;sklearn;matplotlib

**5.说明**

DBN_1Dense.py 构建含一层RBM的信念网络

DBN_2Dense.py 构建含一层RBM+2个全连接层的信念网络

x_DBN_1Dense.py 构建含任意层RBM的深度信念网络

1_Conv_2Dense.py 构建含一层卷积的网络

2_Conv_2Dense.py 构建含二层卷积的网络

5_Conv_WDCNN.py 构建WDCNN模型

noise.py 为数据集添加高斯白噪音函数

preprocess.py 对数据集进行预处理函数

*checkpoint文件夹：保存模型参数，断点续训使用；
data文件夹：CWRU数据集*

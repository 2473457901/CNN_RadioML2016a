# 哈尔滨工业大学(深圳)
# 电信学院：diligent蛋黄派
# 开发时间：2024/1/17 16:41
import time

# 导入相关库
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


# 读取RadioML2016数据集
# RadioML2016.10a包括220000个调制信号，具有20种不同的信噪比（SNR），范围从−20 dB到18 dB，每个调制模式每个SNR 1000个信号。
# 数据集中的每个信号由复数同相和正交（IQ）分量组成。
with open(r'RadioML2016/RML2016.10a_dict.pkl', 'rb') as p_f:
    s = pickle.load(p_f, encoding="latin-1")
# 打印数据
k = 0
for i in s.keys():
    # print(i,s[i])
    # print(s[i])  # 输出字典数据
    print(i)  # 输出数据前类似于('QPSK', 2)的格式
    k = k+1
print(k)

# 判断QPSK的种类个数，一共有SNR=-20~18的，间隔为2dB的QPSK数据
count_qpsk = 0
qpsk_snr_set = set()
for key, _ in s.items():
    if key[0] == 'QPSK':
        count_qpsk += 1
        qpsk_snr_set.add(key[1])
num_qpsk_variations = len(qpsk_snr_set)
print("Number of different 'QPSK' variations:", num_qpsk_variations)
print("Number of 'QPSK' key-value pairs:", count_qpsk)
print("Number of 'QPSK' SNRs of kinds:", qpsk_snr_set)

# 数据集的划分
Xd = pickle.load(open("RadioML2016/RML2016.10a_dict.pkl", 'rb'), encoding='latin')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])  # 取出信噪比和对应的调制方式，并以升序排序
print(snrs, mods)

X = []  # 存储特征数据
lbl = []  # 存储标签数据
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])  # 将对应调制方式和信噪比的特征数据 Xd[(mod, snr)] 添加到列表 X 中
        print(Xd[(mod, snr)].shape)
        # Xd[(mod, snr)].shape[0] 表示特征数据的行数，即样本数量
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))  # 将对应调制方式和信噪比的标签数据 (mod, snr) 添加到列表 lbl 中

# print(X)
# print(lbl)

X = np.vstack(X)

# %%
np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
print(X.shape)  # (220000, 2, 128)
n_examples = X.shape[0]  # 220000
n_train = n_examples * 0.001  # 训练集占比70%
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)  # 在220000个样本中随机不重复选择70%的元素作为训练样本，并取出对应的索引train_idx
test_idx = list(set(range(0, n_examples)) - set(train_idx))  # 在220000个样本中随机选择30%的元素作为训练样本，并取出对应的索引test_idx
X_train = X[train_idx]
X_test = X[test_idx]
X_train = torch.tensor(X[train_idx]).view(len(train_idx), 1, 2, 128)
X_test = torch.tensor(X[test_idx]).view(len(test_idx), 1, 2, 128)

# 调制种类
classes = mods
print(classes)

# 调制种类区分
def classify_mods(output):
    output = output.cpu().detach().numpy()
    idx = np.argmax(output)
    if idx == 0:
        modulation = '8PSK'
    elif idx == 1:
        modulation = 'AM-DSB'
    elif idx == 2:
        modulation = 'AM-SSB'
    elif idx == 3:
        modulation = 'BPSK'
    elif idx == 4:
        modulation = 'CPFSK'
    elif idx == 5:
        modulation = 'GFSK'
    elif idx == 6:
        modulation = 'PAM4'
    elif idx == 7:
        modulation = 'QAM16'
    elif idx == 8:
        modulation = 'QAM64'
    elif idx == 9:
        modulation = 'QPSK'
    elif idx == 10:
        modulation = 'WBFM'

    return modulation, idx


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    yy1 = torch.tensor(yy1)
    return yy1
# 例如对六个状态进行编码：
# 自然顺序码为 000,001,010,011,100,101
# One-Hot编码则是 000001,000010,000100,001000,010000,100000

# trainy 列表中的元素是训练样本对应的调制方式在 mods 列表中的索引
trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = to_onehot(trainy)
print(Y_train)
# testy 列表中的元素是测试样本对应的调制方式在 mods 列表中的索引。
testy = list(map(lambda x: mods.index(lbl[x][0]), test_idx))
Y_test = to_onehot(testy)
print(Y_test)

# 搭建神经网络
class MY_Net(nn.Module):
    def __init__(self):
        super(MY_Net, self).__init__()
        # 原输入图像尺寸为1通道，2 x 128尺寸

        # part-1
        self.pad_1 = nn.ZeroPad2d((2, 2, 0, 0))  # 填充后图像尺寸为1通道，2 x 132尺寸
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=256,
                                kernel_size=(1, 3),
                                stride=1)  # 经过conv_1后图像尺寸变成256通道，2 x 130尺寸
        self.relu_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(p=0.5)

        # part-2
        self.pad_2 = nn.ZeroPad2d((2, 2, 0, 0))  # 填充后图像尺寸为256通道，2 x 134尺寸
        self.conv_2 = nn.Conv2d(in_channels=256,
                                out_channels=80,
                                kernel_size=(2, 3),
                                stride=1)  # 经过conv_2后图像尺寸变成64通道，1 x 132尺寸
        self.relu_2 = nn.ReLU()
        self.drop_2 = nn.Dropout(p=0.5)

        # part-3
        self.linear_1 = nn.Linear(in_features=80 * 1 * 132,
                                  out_features=256)
        self.relu_3 = nn.ReLU()
        self.drop_3 = nn.Dropout(p=0.5)

        # part-4
        self.linear_2 = nn.Linear(in_features=256,
                                  out_features=len(classes))
        self.softmax = nn.Softmax()

    # 前向传播
    def forward(self, x):
        # part-1
        x = self.pad_1(x)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)

        # part-2
        x = self.pad_2(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)

        # part-3
        x = torch.flatten(x)
        x = self.linear_1(x)
        x = self.relu_3(x)
        x = self.drop_3(x)

        # part-4
        x = self.linear_2(x)
        x = self.softmax(x)

        return x

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置批数量
batch_size = 64

# 训练网络实例化
CNN_Model_train = MY_Net().to(device)

# 训练模型构建
def training(epochs):
    CNN_Model_train.train()
    every_epoch_train_loss = []  # 初始化每个轮次的训练损失
    every_epoch_train_accuracy = []  # 初始化每个轮次训练精度
    for epoch in range(epochs):
        every_sample_train_loss = []  # 初始化每个训练样本的训练损失
        print('-----第{}轮训练开始------'.format(epoch+1))
        # 训练过程中的混淆矩阵
        Confusion_Matrix_train = np.empty((11, 11))

        correct = 0  # 初始化每一轮次中的识别正确的个数
        step = 0  # 初试化步数，用于记录每个轮次中处理数据的个数

        start_time_epoch = time.time()  # 每个轮次的开始时间
        for x, y in zip(X_train, Y_train):
            # 初始化数据
            x, y = x.to(device), y.to(device)
            y_predict = CNN_Model_train(x)

            # 调制类别
            modulation_true, idx_true = classify_mods(y)
            modulation_predict, idx_predict = classify_mods(y_predict)

            # 用混淆矩阵记录下参数
            Confusion_Matrix_train[idx_predict, idx_true] += 1

            # 计算是否识别正确
            correct_number = (idx_true == idx_predict).sum()
            correct += correct_number

            # 损失函数
            loss_function = nn.CrossEntropyLoss()
            # 优化器
            optimizer = torch.optim.SGD(CNN_Model_train.parameters(), lr=0.01)
            # 计算每轮损失值
            loss_epoch = loss_function(y_predict, y)
            every_sample_train_loss.append(loss_epoch)

            # 梯度清零
            optimizer.zero_grad()
            loss_epoch.backward()
            optimizer.step()
            # with torch.no_grad():
            #     every_sample_train_loss.append(loss_epoch)

            step += 1

        end_time_epoch = time.time()  # 每个轮次的结束时间

        print('每个轮次的训练数据个数为:{}'.format(step))
        print('本轮次(第{0}轮)训练的准确率:{1}'.format(epoch+1, correct / step))

        # 存储每个轮次的训练精度
        every_epoch_train_accuracy.append(correct / step)
        # 计算每个轮次的平均损失函数值
        average_epoch_loss = sum(every_sample_train_loss) / len(every_sample_train_loss)
        # 存储每个轮次的平均损失函数值
        every_epoch_train_loss.append(average_epoch_loss)

        print('第{0}轮训练集的训练损失为:{1}'.format(epoch+1, every_epoch_train_loss))
        print('第{0}轮训练结束,所用时为:{1}'.format(epoch+1, end_time_epoch - start_time_epoch))

        # 每10个轮次绘制一次混淆矩阵
        if epoch % 1 == 0:
            # 混淆矩阵中的元素转换为整型数据
            Confusion_Matrix_train_picture = Confusion_Matrix_train.astype(int)
            # 定义横纵标签
            x_labels = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QPSK', 'WBFM']
            y_labels = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QPSK', 'WBFM']
            # y_labels = ['WBFM', 'QPSK', 'QAM16', 'PAM4', 'GFSK', 'CPFSK', 'BPSK', 'AM-SSB', 'AM-DSB', '8PSK']
            # 绘制热力图
            plt.imshow(Confusion_Matrix_train_picture, cmap='Blues', interpolation='nearest')
            # 定义横纵坐标刻度与标签
            plt.xticks(np.arange(len(x_labels)), x_labels)
            plt.yticks(np.arange(len(y_labels)), y_labels)
            # 添加热力图颜色渐变条
            plt.colorbar()
            # 添加X轴和Y轴标注
            plt.xlabel('True Labels')
            plt.ylabel('Predicted Labels')
            # 填充混淆矩阵元素
            # 在方格中显示对应的元素值
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    plt.annotate(str(Confusion_Matrix_train_picture[i, j]), xy=(j, i), ha='center', va='center')
            # 显示图形，并暂停10秒后自动关闭
            plt.show(block=False)
            plt.pause(10)
            plt.close()




# 测试网络实例化
CNN_Model_test = MY_Net().to(device)

# 训练模型存储初始化

# 测试模型构建
def testing(epochs):
    CNN_Model_test.train()
    every_epoch_test_loss = []  # 初始化每个轮次的测试损失
    every_epoch_test_accuracy = []  # 初始化每个轮次测试精度
    with torch.no_grad():  # 直接在已训练的模型上引入测试集进行测试，而不引入梯度优化
        for epoch in range(epochs):
            print('-----第{0}轮测试开始------'.format(epoch+1))
            # 测试过程中的混淆矩阵
            Confusion_Matrix_test = np.empty((11, 11))

            correct = 0  # 初始化每一轮次中的识别正确的个数
            step = 0  # 初试化步数，用于记录每个轮次中处理数据的个数
            start_time_epoch = time.time()  # 每个轮次的开始时间
            every_sample_test_loss = []  # 初始化每个训练样本的测试损失
            for x, y in zip(X_test, Y_test):
                # 初始化数据
                x, y = x.to(device), y.to(device)
                y_predict = CNN_Model_test(x)

                # 调制类别
                modulation_true, idx_true = classify_mods(y)
                modulation_predict, idx_predict = classify_mods(y_predict)

                # 用混淆矩阵记录下参数
                Confusion_Matrix_test[idx_true, idx_predict] += 1

                # 计算是否识别正确
                correct_number = (y_predict.argmax(1) == y.argmax(1)).sum()
                correct += correct_number

                # 损失函数
                loss_function = nn.CrossEntropyLoss()

                # 不需要优化器
                # optimizer = torch.optim.SGD(CNN_Model_test.parameters(), lr=0.001)

                # 计算每轮损失值
                loss_epoch = loss_function(y_predict, y)

                # 不需要对梯度操作
                # optimizer.zero_grad()
                # loss_epoch.backward()
                # optimizer.step()

                every_sample_test_loss.append(loss_epoch)
                step += 1

            end_time_epoch = time.time()  # 每个轮次的结束时间

            print('每个轮次的测试数据个数为:{0}'.format(step))
            print('本轮次(第{0})测试的准确率:{1}'.format(epoch+1, correct / step))

            # 存储每个轮次的测试精度
            every_epoch_test_accuracy.append(correct / step)
            # 计算每个轮次的测试平均损失函数值
            average_epoch_loss = sum(every_sample_test_loss) / len(every_sample_test_loss)
            # 存储每个轮次的测试平均损失函数值
            every_epoch_test_loss.append(average_epoch_loss)

            print('第{0}轮测试集的训练损失为:{1}'.format(epoch+1, every_epoch_test_loss))
            print('第{0}轮测试结束,所用时为:{1}'.format(epoch+1, end_time_epoch - start_time_epoch))

            # 每10个轮次绘制一次混淆矩阵
            if epoch % 1 == 0:
                # 定义横纵标签
                x_labels = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QPSK', 'WBFM']
                y_labels = ['WBFM', 'QPSK', 'QAM16', 'PAM4', 'GFSK', 'CPFSK', 'BPSK', 'AM-SSB', 'AM-DSB', '8PSK']
                # 绘制热力图
                plt.imshow(Confusion_Matrix_test, cmap='Blues', interpolation='nearest', alpha=0.5)
                # 定义横纵坐标刻度与标签
                plt.xticks(np.arange(len(x_labels), x_labels))
                plt.yticks(np.arange(len(y_labels), y_labels))
                # 添加热力图颜色渐变条
                plt.colorbar()
                # 添加X轴和Y轴标注
                plt.xlabel('True Labels')
                plt.ylabel('Predicted Labels')
                # 显示热力图
                plt.show()



# 创建实例化模型
All_epochs = 100  # 训练或测试的总轮次
training(epochs=All_epochs)






















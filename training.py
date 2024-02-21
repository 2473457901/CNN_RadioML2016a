# 哈尔滨工业大学(深圳)
# 电信学院：diligent蛋黄派
# 开发时间：2024/2/14 11:01

import pickle
import time
import matplotlib.pyplot as plt
from BaseModule import *
from classify_mods import *
from To_OneHot import *


# 设置批数量
batch_size = 64

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 训练网络实例化
CNN_Model_train = MY_Net().to(device)
# 训练模型构建
def training(epochs, X_train, Y_train):
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
# 哈尔滨工业大学(深圳)
# 电信学院：diligent蛋黄派
# 开发时间：2024/2/14 11:02

import torch
from torch import nn
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
                                  out_features=11)
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
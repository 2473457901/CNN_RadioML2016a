# 哈尔滨工业大学(深圳)
# 电信学院：diligent蛋黄派
# 开发时间：2024/2/14 11:22


# 导入相关库
import torch
import pickle
import numpy as np
from training import training
from To_OneHot import to_onehot
from preprocessing_data import *

# 创建实例化模型
All_epochs = 100  # 训练或测试的总轮次
training(epochs=All_epochs, X_train=X_train_new, Y_train=Y_train_new)
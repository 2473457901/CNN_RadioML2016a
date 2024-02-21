# 哈尔滨工业大学(深圳)
# 电信学院：diligent蛋黄派
# 开发时间：2024/2/14 11:12

import torch
import numpy as np
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    yy1 = torch.tensor(yy1)
    return yy1
# 例如对六个状态进行编码：
# 自然顺序码为 000,001,010,011,100,101
# One-Hot编码则是 000001,000010,000100,001000,010000,100000


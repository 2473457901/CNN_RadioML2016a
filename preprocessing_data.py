# 哈尔滨工业大学(深圳)
# 电信学院：diligent蛋黄派
# 开发时间：2024/2/21 20:55

# 读取RadioML2016数据集
# RadioML2016.10a包括220000个调制信号，具有20种不同的信噪比（SNR），范围从−20 dB到18 dB，每个调制模式每个SNR 1000个信号。
# 数据集中的每个信号由复数同相和正交（IQ）分量组成。
import pickle
from To_OneHot import *


with open(r'RadioML2016/RML2016.10a_dict.pkl', 'rb') as p_f:
    s = pickle.load(p_f, encoding="latin-1")
#
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
X_train_new = torch.tensor(X[train_idx]).view(len(train_idx), 1, 2, 128)
X_test_new = torch.tensor(X[test_idx]).view(len(test_idx), 1, 2, 128)

# 调制种类
classes = mods
print(classes)

# trainy 列表中的元素是训练样本对应的调制方式在 mods 列表中的索引
trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train_new = to_onehot(trainy)
print(Y_train_new)
# testy 列表中的元素是测试样本对应的调制方式在 mods 列表中的索引。
testy = list(map(lambda x: mods.index(lbl[x][0]), test_idx))
Y_test_new = to_onehot(testy)
print(Y_test_new)
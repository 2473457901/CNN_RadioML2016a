# 哈尔滨工业大学(深圳)
# 电信学院：diligent蛋黄派
# 开发时间：2024/2/14 11:03

import numpy as np
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
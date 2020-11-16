# 生成数据文件——10000组包含1000个点的数据集
import numpy as np
from time import time
import csv
import time

section = np.linspace(-3, 3, 100)
section = section.tolist()

def plot_cloud_model(Ex, En, He, n):
    # Ex = 0      # 期望
    # En = 1      # 熵
    # He = 0.25   # 超熵
    # n = 1000     # 云滴个数

    Y = np.zeros((99, n))
    np.random.seed(int(time.time()))
    X = np.random.normal(loc=En, scale=He, size=n)
    Y = Y[0]  # 什么作用？？？

    temp = np.zeros((100,1))

    for i in range(n):
        np.random.seed(int(time.time()) + i + 1)
        Enn = X[i]
        X[i] = np.random.normal(loc=Ex, scale=np.abs(Enn), size=1)

        for j in range(100):
            if j < 99:
                if X[i] >= section[j] and X[i] < section[j+1]:
                    temp[j][0] += 1
                    break

    temp = temp.tolist()
    count = []
    for i in range(100):
        count.append(temp[i][0])
    return count


f = open("test_case.csv", 'w', newline="")
writer = csv.writer(f)
for i in range(10):
    temp_cout = plot_cloud_model(0,1,0.25,1000)
    print(temp_cout)
    writer.writerow(temp_cout)
    time.sleep(1)
f.close()


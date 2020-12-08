# 生成数据文件——10000组包含1000个点的数据集
import numpy as np
from time import time
import csv
import time
import random

def plot_cloud_model(Ex, En, He, n):
    # Ex = 0      # 期望
    # En = 1      # 熵
    # He = 0.25   # 超熵
    # n = 1000     # 云滴个数

    Y = np.zeros((99, n))
    np.random.seed(int(time.time()))
    X = np.random.normal(loc=En, scale=He, size=n)
    Y = Y[0]

    temp = np.zeros((100,1))

    # topEdge = En + 3 * He
    # buttonEdge = En - 3 * He
    # section = np.linspace(buttonEdge, topEdge, 100)
    # section = section.tolist()

    for i in range(n):
        np.random.seed(int(time.time()) + i + 1)
        Enn = X[i]

        X[i] = np.random.normal(loc=Ex, scale=np.abs(Enn), size=1)

        # for j in range(100):
        #     if j < 99:
        #         if X[i] >= section[j] and X[i] < section[j+1]:
        #             temp[j][0] += 1
        #             break

    # temp = temp.tolist()
    # count = []
    # for i in range(100):
    #     count.append(temp[i][0])
    return X


f1 = open("training_case_3_output.csv", 'w', newline="")
f1_1 = open("training_case_3_output_label.csv", 'w', newline="")
writer1 = csv.writer(f1)
writer1_1 = csv.writer(f1_1)
for i in range(900):
    temp_cout = plot_cloud_model(0,1,0.25,100)
    writer1.writerow(temp_cout)
    writer1_1.writerow([0,1,0.25])
    time.sleep(1)

for i in range(100):
    temp_en = random.uniform(10,20)
    temp_he = random.uniform(10,20)
    temp_cout = plot_cloud_model(0,temp_en,temp_he,100)
    writer1.writerow(temp_cout)
    writer1_1.writerow([0, temp_en, temp_he])

f1.close()

f2 = open("test_case_3_output.csv", 'w', newline="")
writer2 = csv.writer(f2)
for i in range(10):
    temp_cout = plot_cloud_model(0,1,0.25,100)
    writer2.writerow(temp_cout)
    time.sleep(1)

for i in range(10):
    temp_cout = plot_cloud_model(0,random.uniform(10,20),random.uniform(10,20),100)
    writer2.writerow(temp_cout)

f2.close()


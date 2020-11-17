import tensorflow as tf
import numpy as np
import csv

input_x = []
f1 = csv.reader(open('training_case.csv','r'))
for index, i in enumerate(f1):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    # print(len(temp_x))
    input_x.insert(index, temp_x)

all_y_trues = [] # 对应输入数据的预期 输出向量
for i in range(len(input_x)):
    if(i < 10):
        all_y_trues.insert(i, [1])
    else:
        all_y_trues.insert(i, [0])

test_case = []
f2 = csv.reader(open('test_case.csv','r'))
for index, i in enumerate(f2):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    # print(len(temp_x))
    test_case.insert(index, temp_x)

inputs = tf.keras.Input(shape=(100,))
x = tf.keras.layers.Dense(201, use_bias=True, activation='sigmoid')(inputs)
outputs = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(x)
m = tf.keras.Model(inputs, outputs)     # 使用 输入 和 输出 创建模型

m.compile(tf.keras.optimizers.SGD(learning_rate=0.1), 'mse')
m.fit(input_x, all_y_trues, epochs=1000, batch_size=1, verbose=0)

for i in range(len(test_case)):
    temp_case = np.array([test_case[i]])
    print(temp_case)
    print("Case" + str(i) + ": " + str(m.predict(temp_case)) )

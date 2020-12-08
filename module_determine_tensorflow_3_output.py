import tensorflow as tf
import numpy as np
import csv

input_x = []
f1 = csv.reader(open('training_case_3_output.csv','r'))
for index, i in enumerate(f1):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    input_x.insert(index, temp_x)

lables = [] # 对应输入数据的预期 输出向量
f1_1 = csv.reader(open('training_case_3_output_label.csv','r'))
for index, i in enumerate(f1_1):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append((float(i[elm])))
    lables.append(temp_x)

test_case = []
f2 = csv.reader(open('test_case_3_output.csv','r'))
for index, i in enumerate(f2):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    test_case.insert(index, temp_x)

inputs = tf.keras.Input(shape=(100,))
x1 = tf.keras.layers.Dense(201, use_bias=True, activation='sigmoid')(inputs)
x2 = tf.keras.layers.Dense(201, use_bias=True, activation='relu')(x1)
x3 = tf.keras.layers.Dense(201, use_bias=True, activation='sigmoid')(x2)
outputs = tf.keras.layers.Dense(3, use_bias=True, activation='relu')(x3)
m = tf.keras.Model(inputs, outputs)     # 使用 输入 和 输出 创建模型

m.compile(tf.keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6), loss = 'mse',metrics = ['accuracy'])
m.fit(input_x, lables, epochs=1000, batch_size=1, verbose=1)

# 保存模型
m.save('3_output_all_x.h5')

for i in range(len(test_case)):
    temp_case = np.array([test_case[i]])
    print("Case" + str(i) + ": " + str(m.predict(temp_case)) )

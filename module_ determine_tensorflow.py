import tensorflow as tf
import numpy as np
import csv

input_x = np.array()
f = csv.reader(open('test_case.csv','r'))
for index, i in enumerate(f):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    print(temp_x)
    input_x.insert(index, temp_x)

print(input_x)     # 处理后的输入矩阵

all_y_trues = [[1]*100] # 对应输入数据的预期 输出向量
print(all_y_trues)

#
# data = np.array([
#   [-2.0, -1],  # Alice
#   [25, 6],   # Bob
#   [17, 4],   # Charlie
#   [-15, -6], # Diana
# ])
# all_y_trues = np.array([
#   1, # Alice
#   0, # Bob
#   0, # Charlie
#   1, # Diana
# ])
#
# inputs = tf.keras.Input(shape=(2,))
# x = tf.keras.layers.Dense(201, use_bias=True, activation='sigmoid')(inputs)
# outputs = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(x)
# m = tf.keras.Model(inputs, outputs)     # 使用 输入 和 输出 创建模型
#
# m.compile(tf.keras.optimizers.SGD(learning_rate=0.1), 'mse')
# m.fit(data, all_y_trues, epochs=1000, batch_size=1, verbose=0)
#
# emily = np.array([[-7, -3]])
# frank = np.array([[20, 2]])
# print(m.predict(emily))
# print(m.predict(frank))
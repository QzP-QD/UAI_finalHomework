import tensorflow as tf
import numpy as np
import csv

# 恢复模型
restored_model = tf.keras.models.load_model('3_output_all_x.h5')

# 读取测试集
test_case = []
f2 = csv.reader(open('test_case_3_output_split.csv','r'))
for index, i in enumerate(f2):
    temp_x = []
    for elm in range(len(i)):
        temp_x.append(int(float(i[elm])))
    test_case.insert(index, temp_x)


for i in range(len(test_case)):
    temp_case = np.array([test_case[i]])
    temp_case = restored_model.predict(temp_case).ravel()
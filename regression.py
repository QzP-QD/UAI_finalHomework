import numpy as np
import matplotlib.pyplot as plt

#房价预测（房子面积）
#x存储输入数据，y存储标准输出
x, y = [], []
# 取出数据集
for sample in open("./prices.txt"):
    _x, _y = sample.split(",")
    x.append(float(_x))
    y.append(float(_y))

#转化为数组，方便处理
x, y = np.array(x), np.array(y)
#数据标准化（降低数据复杂度）
x = (x - x.mean())/x.std()
#画出原始数据（散点图）
# plt.figure()
# plt.scatter(x,y)
# plt.show()

# 选择&训练模型
# 在（-2，4）取100个点作为画图基础点
x0 = np.linspace(-2,4,100)
# 利用Numpy的函数定义巡礼啊并返回回归模型的函数
# deg代表参数中n/多项式次数
# 返回的模型能够根据输入x0，返回预测的y
def get_model(deg):
    # lambda是“匿名函数”
    return lambda input_x=x0 : np.polyval(np.polyfit(x,y,deg), input_x)

def get_cost(deg, input_x, inpit_y):
    return 0.5
import numpy as np

'''
通过身高体重估计性别
神经网络
    -一个两个神经元的隐藏层(h1,h2)
    -两个输入
    - 一个神经元的输出层(o1)
'''
# 激活函数sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# sigmoid函数的导数：f'(x) = f(x) * (1 - f(x))
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)

# MSE损失：y_true和y_false均为向量
def mse_loss(y_true, y_pred):
    return ((y_true-y_pred)**2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # 权重向量乘输入向量，加上偏置，然后套用sigmoid
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class OurNeuralNetwork:

    '''
    初始化，设置权重值和偏置量
    '''
    def __init__(self):
        # 权重，Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        #截距，bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    '''
    前向传播，得出预测的结果值
    '''
    def feedfoward(self,x):
        # x是输入向量 （2个元素）
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1;

    def train(self,data,all_y_trues):
        '''
        data是训练集（n*2）矩阵，n是样本数量
        all_y_trues是n维向量，表示真实的输出值
        '''
        learn_rate = 0.1 # 学习率
        epochs = 1000 # 循环训练集的次数 ?

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues): # zip()打包成元组的列表
                # 进行前向传播（预测）——后面反向传播需要用到这些值，所以都算出来
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                '''
                计算偏导数
                命名规则：d_L_d_w1 意为 “L偏导w1”
                '''
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h2)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # ————更新 权重 和 偏移量
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # 在每次循环结束计算总 损失值
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedfoward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss : %.3f" % (epoch, loss))

# 设定数据集
data = np.array([
    [-2,-1],    #ALice
    [25,6],     #Bob
    [17,4],     #Charlie
    [-15, -6],  #Diana
])

all_y_trues = np.array([
    1,          #Alice
    0,          #Bob
    0,          #Charlie
    1           #Diana
])

# 训练神经网络
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedfoward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedfoward(frank)) # 0.039 - M
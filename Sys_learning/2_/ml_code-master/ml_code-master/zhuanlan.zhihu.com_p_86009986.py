import numpy as np
import matplotlib.pyplot as plt


def generate(): #生成X，y的列向量，x从50到150，y = 3.6x+噪声N（0，30）
    x = np.linspace(50, 150)
    w = 3.6
    y = w * x + np.random.normal(0, 30, size=x.size)
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    return X, Y


class LinearRegression(object):#线性回归类
    def __init__(self, w=None):
        self.w = w

    def least_square(self, X, Y):
        '''
        最小二乘法，通过给定样本学习参数
        X: 样本矩阵，一行一样本
        Y: 样本标签，Nx1矩阵
        '''
        inv = np.linalg.inv(np.dot(X.T, X)) #矩阵求逆
        self.w = inv.dot(np.dot(X.T, Y))

    def ridge(self, X, Y, lam=3e-2):
        # 岭回归
        inv = np.linalg.inv(np.dot(X.T, X) + np.diag([lam] * X.shape[-1]))
        self.w = inv.dot(np.dot(X.T, Y))

    def predict(self, x):
        '''
        通过给定数据进行预测
        '''
        return np.dot(x, self.w)


class BayesRegression(object):#贝叶斯回归类
    def __init__(self, sigma, mu=None, cov=None):
        self.sigma = sigma  # 噪声方差
        self.mu = mu  # 后验分布的均值
        self.cov = cov  # 后验分布的协方差矩阵

    def fit(self, X, Y):
        '''
        拟合后验参数
        '''
        prior = np.eye(X.shape[-1])  # 先验协方差
        A = np.dot(X.T, X) / self.sigma
        self.cov = np.linalg.inv(A)
        self.mu = self.cov.dot(np.dot(X.T, Y)) / self.sigma

    def generate(self, x):
        '''
        根据给定数据，生成预测值
        '''
        mean = np.dot(x, self.mu).sum()
        std = np.sqrt(np.dot(x, self.cov).dot(x)).sum()
        return np.random.normal(mean, std, 1)

    def generate_random(self, x):
        '''
        根据给定数据，生成带噪声的预测值
        '''
        mean = np.dot(x, self.mu).sum()
        std = np.sqrt(np.dot(x, self.cov).dot(x) + self.sigma).sum()
        return np.random.normal(mean, std, 1)


def main():
    X, Y = generate()
    lr = LinearRegression()
    lr.ridge(X, Y)
    x = np.array([50, 150]).reshape(-1, 1)
    y = lr.predict(x)

    plt.figure()
    plt.scatter(X, Y)
    plt.plot(x, y, c='r')
    for i, j in zip(X, Y):
        plt.plot([i, i], [j, lr.predict(i)], c='b', linestyle='dotted')
    plt.show()


if __name__ == "__main__":
    # main()
    X, Y = generate()
    lr = LinearRegression()
    lr.least_square(X, Y)
    sigma = np.square(Y - np.dot(X, lr.w)).mean()
    '''
    关于贝叶斯线性回归需要将噪声的方差传给模型，但是计算方差又需要具体的w值
    所以没办法只能通过判别模型的线性回归拟合w，然后再计算噪声的方差
    '''
    br = BayesRegression(sigma)
    br.fit(X, Y)

    xx = []
    yy = []
    count = 30
    for x in np.linspace(50, 150):
        xx += [x] * count
        yy += [br.generate_random(x) for _ in range(count)]

    #plt.figure(figsize=(21, 9))
    #plt.scatter(xx, yy, c='r')
    #plt.show()
   
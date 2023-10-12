from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import Ridge

X, y = load_boston(return_X_y=True)
trainx, testx, trainy, testy = train_test_split(X, y, test_size = 0.33, random_state = 42)
# trainx, trainy = trainx[:50], trainy[:50]

def linear_regression(X_train, y_train):
    ''' 线性回归
    :参数 X_train: np.ndarray, 形状为(n, d), n个d维训练样本
    :参数 y_train: np.ndarray, 形状为(n, 1), 每个样本的标签
    :返回: 权重矩阵W
    '''
    [n, d] = X_train.shape
    ones = np.ones((n, 1))
    X = np.hstack((X_train, ones))
    Xt = np.transpose(X)
    XtX = Xt @ X
    Xty = Xt @ y_train
    W = np.linalg.solve(XtX, Xty)
    return W

def ridge_regression(X_train, y_train, lmbd):
    ''' 岭回归
    :参数 X_train: np.ndarray, 形状为(n, d), n个d维训练样本
    :参数 y_train: np.ndarray, 形状为(n, 1), 每个样本的标签
    :参数 lmbd: float, 岭回归lambda参数
    :返回: 权重矩阵W
    '''
    [n, d] = X_train.shape
    ones = np.ones((n, 1))
    eye = np.eye(d + 1)
    eye[d, d] = 0
    X = np.hstack((X_train, ones))
    Xt = np.transpose(X)
    XtX = Xt @ X + 2 * lmbd * eye
    Xty = Xt @ y_train
    W = np.linalg.solve(XtX, Xty)
    return W

def MSE(X_train, y_train, X_test, y_test, lmbd=None):
    ''' 计算MSE, 根据lmbd是否输入判断是否岭回归
    :参数 X_train: np.ndarray, 形状为(n, d), n个d维训练样本
    :参数 y_train: np.ndarray, 形状为(n, 1), 每个训练样本的标签
    :参数 X_test: np.ndarray, 形状为(m, d), m个d维测试样本
    :参数 y_test: np.ndarray, 形状为(m, 1), 每个测试样本的标签
    :参数 lmbd: float或None, 岭回归\lambda参数, None表示使用线性回归
    :返回: 标量, MSE值
    '''
    W = linear_regression(X_train, y_train) if lmbd is None else ridge_regression(X_train, y_train, lmbd)
    m = X_test.shape[0]
    ones = np.ones((m, 1))
    X = np.hstack((X_test, ones))
    E = y_test - X @ W
    return np.mean(E ** 2)
    

# 针对基本线性回归和岭回归模型计算MSE
linear_regression_MSE= lambda : MSE(trainx, trainy, testx, testy)
ridge_regression_MSE= lambda lmbd : MSE(trainx, trainy, testx, testy, lmbd)

lr = linear_regression_MSE()
rr1 = ridge_regression_MSE(1)

print(lr, rr1)

ridge_sklearn = Ridge(alpha = 2).fit(trainx, trainy)
r2_score = ridge_sklearn.score(testx, testy)
predicty = ridge_sklearn.predict(testx)
print(np.mean((predicty - testy) ** 2))
print(r2_score)
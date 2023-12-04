import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cp

def load_data():
    mean1 = [-3, -3]
    sigma1 = [[2, -1], [-1, 2]]
    mean2 = [3, 3]
    sigma2 = [[2, -1], [-1, 2]]
    X1 = np.random.multivariate_normal(mean1, sigma1, 100)
    X2 = np.random.multivariate_normal(mean2, sigma2, 100)
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(100), -np.ones(100)))
    return X, y

def fit(X, y):
    x = X.copy()
    x[y == -1, :] = -x[y == -1, :]
    n = x.shape[0]
    alpha = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(x.T @ alpha) - np.ones(n) @ alpha)
    constraint = [alpha >= 0, y @ alpha == 0]
    prob = cp.Problem(objective, constraint)
    prob.solve(solver='CVXOPT')
    print(f"dual variable = {alpha.value}")
    w = x.T @ alpha.value
    index = np.where(abs(alpha.value / max(alpha.value)) > 1e-3)[0]
    print(f"support vector index = {index}")
    b = np.mean(y[index] - X[index, :] @ w)
    return w, b, index
    
if __name__=="__main__":
    X, y = load_data()
    w, b, index = fit(X, y)
    print(f"w = {w}, b = {b}")
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.scatter(X[y == -1, 0], X[y == -1, 1])
    plt.scatter(X[index, 0], X[index, 1], marker='*', s=100)
    plt.plot((-4, 4), ((-b + 4 * w[0]) / w[1], (-b - 4 * w[0]) / w[1]))
    plt.show()
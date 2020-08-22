import numpy as np

class LinearRegression:

    # lr is the learning rate, n_iters the iterations to get to a minimum error
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None # angular coefficients
        self.bias = None # q, the y offset

    def fit(self, X, y):
        # gradient descent:
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) 
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples)*np.dot(X.T, (y_predicted - y))  # derivative with respect to m
            db = (1/n_samples)*np.sum(y_predicted-y) # derivative of q (in y=mx+q)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
        

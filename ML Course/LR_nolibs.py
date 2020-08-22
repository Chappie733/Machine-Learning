def dot(a,b):
    result = 0
    if type(a) == int:
        for i in range(len(b)):
            result += a*b[i]
    else:
        for i in range(len(a)):
            result += a[i]*b[i]
    return result

class LinearRegression:

    def __init__(self, alpha=0.01, n_iters = 1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.m = None # angular coefficient (there's one for each feature in the inputs)
        self.q = 0 # y offset/known term

    def fit(self, X, y):
        n = len(X)
        self.m = [0 for _ in X[0]]
        
        for _ in range(self.n_iters):
            mse_tot_m = 0 # summatory of m in its derivative
            mse_tot_q = 0 # summatory of q in its derivative
            # calculate summatories
            for i in range(n):
                y_predicted = dot(self.m,X[i])+self.q
                y_true = y[i]
                mse_tot_m += dot(-2, X[i])*(y_true-y_predicted)
                mse_tot_q += -2*(y_true-y_predicted)
            # find actual derivative
            dm = (1/n)*mse_tot_m
            dq = (1/n)*mse_tot_q

            self.m -= self.alpha*dm
            self.q -= self.alpha*dq

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        return dot(self.m, x)+self.q

from math import sqrt

def euclidean_distance(a,b):
    res = 0
    for i in range(len(a)):
        res += (a[i]-b[i])**2
    return sqrt(res)


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, X):
        predicted = [ self.y_train[self._predict(x)] for x in X ]
        return predicted

    def _predict(self, x):
        closest = []
        for index in range(len(self.x_train)):
            if len(closest) < self.k:
                closest.append(index)
            else:
                for i in range(len(closest)):
                    if euclidean_distance(self.x_train[closest[i]], x) > euclidean_distance(self.x_train[i], x):
                        closest[i] = index
                        break
        return closest

#given an input x, we have as a result the output y of the closest
#x saved as in the training set, (Knn) is for k-nearest-neighbors

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2)) # uses numpy but follows the euclidean distance formula
    
class KNN:

    def __init__(self, k=3):
        # k is the number of nearest neighbors we consider for an output
        self.k = k

    # x are inputs, y are outputs
    def fit(self, x,y):
        self.X_train = x # store the inputs
        self.Y_train = y # store the outputs

    # can have multiple samples
    def predict(self, x):
        predicted_labels = [self._predict(s) for s in x] # outputs
        return np.array(predicted_labels) # return as a numpy array
        
    # only one sample
    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train] # gets all the distances
        
        # get k-nearest neighbors and labels
        k_indices = np.argsort(distances)[:self.k] # gets the indices of the closest k samples
        k_nearest_labels = [self.Y_train[i] for i in k_indices] # outputs of nearest neighbors
        
        # majority vote (if there are 2 samples of type A and 1 of type B, choose type A as a label)
        most_common = Counter(k_nearest_labels).most_common(1) # final output (1) means "get only the most common"
        return most_common[0][0]

import numpy as np
class LinearRegression:
    def __init__(self,lr=0.001,n_iteration=1000):
        self.lr=lr
        self.n_iteration=n_iteration
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_samples,n_features=X.shape()
        self.weights=np.zeros(n_features)
        self.bias=0

        for i in range(n_iteration):
            y_pred = np.dot(X.T,self.weights)+self.bias

            dw=(1/n_samples)*np.dot(X,(y_pred-y))
            db=(1/n_samples)*np.sum(y_pred-y)

            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db

    def predict(self,X):
        y_pred=np.dot(X,self.weights)+self.bias
        return y_pred
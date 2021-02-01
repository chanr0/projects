import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope

from sklearn.svm import OneClassSVM


def remove_outlier_OCSVM(X,X_test,y):
    lof = OneClassSVM(nu=0.02, gamma=0.001)
    lof.fit(X)
    yhat = lof.predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X, y = X[mask, :], y[mask]
    return X, y 





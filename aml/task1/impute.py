import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer



def impute_data(X,strategy='median', X_test = None, k=4):
    if strategy =='knn':
        return impute_knn(X=X, X_test=X_test, k=k)
    else:
        # imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        # # impute trainig data
        # X = imp.fit_transform(X)
        # if X_test is not None:
        #     X_test = imp.fit_transform(X_test)
        #     return X, X_test
        # else:
        #     return X
        X_tmp = X
        X_nrows = X_tmp.shape[0]
        
        if X_test is not None:
            X_test_nrows = X_test.shape[0]
            X_tmp = np.concatenate((X,X_test))
        
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        X_tmp = imputer.fit_transform(X_tmp)
        if X_test is not None:
            X_new = X_tmp[0:X_nrows,:]
            X_test_new = X_tmp[X_nrows:,:]
            return X_new, X_test_new
        else:
            return X_tmp


def impute_knn(X, X_test, k=3):
    X_tmp = X
    X_nrows = X_tmp.shape[0]
    
    if X_test is not None:
        X_test_nrows = X_test.shape[0]
        X_tmp = np.concatenate((X,X_test))
    
    imputer = KNNImputer(n_neighbors=k)
    X_tmp = imputer.fit_transform(X_tmp)
    if X_test is not None:
        X_new = X_tmp[0:X_nrows,:]
        X_test_new = X_tmp[X_nrows:,:]
        return X_new, X_test_new
    else:
        return X_tmp


    





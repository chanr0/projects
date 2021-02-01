import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from OCSVM import remove_outlier_OCSVM
from impute import impute_data

################### Define paths ###################
base_path = ""
X_train_data = base_path + "data/X_train.csv"
y_train_data = base_path + "data/y_train.csv"
X_test_data = base_path + "data/X_test.csv"
submisson_file = base_path + "submissions/submisson_med_stand_kbest_ocsvm_svr_2.csv"

################### Load data and format ###################
# load training data dataset
df_X_train = pd.read_csv(X_train_data, float_precision = "round_trip")
df_y_train = pd.read_csv(y_train_data, float_precision = "round_trip")

# load test data dataset
df_X_test = pd.read_csv(X_test_data, float_precision = "round_trip")
ex_ID = df_X_test.get('id')  # ids for export

# create numpy arrays
X = (df_X_train.values)[:, 1:]
y = (df_y_train.values)[:, 1]

X_test = (df_X_test.values)[:, 1:]

################### Impute missing data with column median ###################
X, X_test = impute_data(X, strategy = "median", X_test = X_test)
print(f'X.shape = {X.shape}; y.shape = {y.shape}; X_test.shape = {X_test.shape}')

################### Standard Scaling/ Normalizing ###################
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

################### Remove Highly Correlated Features ######################

df = pd.DataFrame(X)

# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

df.drop(df[to_drop], axis = 1)
X = df.to_numpy()

df = pd.DataFrame(X_test)
df.drop(df[to_drop], axis = 1)
X_test = df.to_numpy()

################### SelectK best ###################
filter_k = SelectKBest(f_regression, k = "all")
filter_k.fit(X, y)

to_drop = []
for index in range(X.shape[1]):
    if filter_k.pvalues_[index] >= 0.0005:  # don't keep feature
        to_drop.append(index)

X = np.delete(X, to_drop, 1)
X_test = np.delete(X_test, to_drop, 1)

print(f'X.shape = {X.shape}; y.shape = {y.shape}; X_test.shape = {X_test.shape}')

################### One Class SVM (Outlier detection)###################
X, y = remove_outlier_OCSVM(X, X_test, y)
print(f'X.shape = {X.shape}; y.shape = {y.shape}; X_test.shape = {X_test.shape}')

################### SVR ###################

parameters = { 'C': np.arange(80, 91, 2)
               }

clf = GridSearchCV(estimator = SVR(),
                   param_grid = parameters,
                   cv = 10,
                   verbose = 1,
                   n_jobs = -1,
                   scoring = "r2")

clf.fit(X, y)
dict_clf_best_params = clf.best_params_
print(dict_clf_best_params)
print(clf.best_score_)

best_C = dict_clf_best_params["C"]

################### Validate score ###################
# scores = np.ones(3000)
# for i in range(3000):
#     X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
#     reg = SVR(C= best_C)
#     reg.fit(X_tr,y_tr)
#     y_hat = reg.predict(X_te)
#     scores[i] = r2_score(y_te,y_hat)


# import matplotlib.pyplot as plt
# import math

# plt.hist(scores,bins = 300, density=True)
# plt.xlabel('Scores')
# plt.title(f'Histogram of Scores for SVR(C={best_C}) with preprocessed data')
# plt.show()


################### Export Results ###################
y_pred = clf.predict(X_test)
ex_test = pd.DataFrame(data = y_pred, columns = ['y'])
export = pd.concat([ex_ID, ex_test], axis = 1)
export.to_csv(submisson_file, encoding = 'utf-8', index = False)

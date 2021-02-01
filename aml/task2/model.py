import os

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.svm import SVC


class TrainingEngine:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __call__(self, trial):

        #################################
        # CREATE DATAFRAME COPY
        #################################

        X_train_copy = self.X_train.copy()
        y_train_copy = self.y_train.copy()

        NUMBER_SPLITS = 5

        stratified_split = StratifiedShuffleSplit(n_splits = NUMBER_SPLITS, test_size = 0.2)

        final_score = 0
        for train_index, cv_index in stratified_split.split(X_train_copy, y_train_copy):
            X_cross, y_cross = X_train_copy[cv_index], y_train_copy[cv_index]
            X_train, y_train = X_train_copy[train_index], y_train_copy[train_index]

            #################################
            # STANDARDISE
            #################################

            standardise = trial.suggest_categorical('standardise',
                                                    ['None'])
            if standardise == 'StandardScaler':
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_cross = scaler.transform(X_cross)
            elif standardise == 'L2Norm':
                X_train, norms = normalize(X_train, norm = 'l2', axis = 0, return_norm = True)
                for index, norm in enumerate(norms):
                    X_cross[:, index] = X_cross[:, index] / norm
            elif standardise == 'L1Norm':
                X_train, norms = normalize(X_train, norm = 'l1', axis = 0, return_norm = True)
                for index, norm in enumerate(norms):
                    X_cross[:, index] = X_cross[:, index] / norm
            elif standardise == 'MinMax':
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_cross = scaler.transform(X_cross)

            #################################
            # CHOOSE MODEL
            #################################
            model_type = trial.suggest_categorical('model_type', ['SVC'])

            if model_type == 'SVC':
                #################################
                # DEFINE MODEL
                #################################

                kernel = trial.suggest_categorical('SVC_kernel',
                                                   ['poly', 'rbf', 'sigmoid'])
                if kernel == 'poly':
                    degree = trial.suggest_int('SVC_degree', 0, 10)
                else:
                    degree = 3

                if kernel == 'poly' or kernel == 'sigmoid':
                    coef0 = trial.suggest_uniform('SVC_coef0', 0, 1)
                else:
                    coef0 = 0

                # clf = SVC(
                #     gamma = trial.suggest_categorical('SVC_gamma', [2 ** k for k in range(-15, 4)]),
                #     class_weight = trial.suggest_categorical('SVC_class_weight',
                #                                              ['balanced']),
                #     C = trial.suggest_categorical('SVC_C', [2 ** k for k in range(-5, 15)]),
                #     degree = degree,
                #     kernel = kernel,
                #     coef0 = coef0,
                #     probability = True,
                #     shrinking = trial.suggest_categorical('SVC_shrinking', [True, False]),
                #     tol = 0.001,
                #     verbose = False,
                #     break_ties = True
                # )
                ""
                "{'stand_norm': None, 'model_type': 'SVC', 'SVC_gamma': 0.0014989020723407036, 'SVC_class_weight': 'balanced', 'SVC_C': 0.9084772978383183, 'SVC_kernel': 'rbf', 'SVC_shrinking': False, 'SVC_probability': False, 'SVC_break_ties': True}"
                clf = SVC(
                    gamma = 0.8712506123230085,
                    class_weight = 'balanced',
                    C = 2.342911662498267,
                    degree = 2,
                    kernel = 'poly',
                    coef0 = 0.6663647769478728,
                    probability = True,
                    shrinking = False,
                    tol = 0.001,
                    verbose = False,
                    break_ties = True
                )

                #################################
                # TRAIN MODEL
                #################################
                clf.fit(X_train, y_train)
                score = balanced_accuracy_score(y_cross, clf.predict(X_cross))

                final_score += (score / NUMBER_SPLITS)

            if model_type == 'LGBM':
                #################################
                # CONVERT OUTPUT TO ONE HOT ENCODING
                #################################

                # aux = np.zeros((y_train.size, y_train.max() + 1))
                # aux[np.arange(y_train.size), y_train] = 1
                # y_train = aux.copy()
                #
                # aux = np.zeros((y_cross.size, y_cross.max() + 1))
                # aux[np.arange(y_cross.size), y_cross] = 1
                # y_cross = aux.copy()

                # setting up the parameters

                d_train = lgb.Dataset(X_train, label = y_train)
                lgb_classifier = lgb.LGBMClassifier(
                    boosting_type = 'gbdt',
                    learning_rate = trial.suggest_categorical('LGBM_learning_rate',
                                                              [1e-1, 1e-2, 1e-3, 2e-1, 2e-2, 2e-3, 5e-1, 5e-2, 5e-3]),
                    objective = trial.suggest_categorical('LGBM_objective', ['multiclass']),
                    max_depth = -1,
                    class_weight = 'balanced',
                    n_estimators = 10000,
                    num_leaves = trial.suggest_int('LGBM_num_leaves', 6, 50),
                    min_child_samples = trial.suggest_int('LGBM_min_child_samples', 100, 500),
                    min_child_weight = trial.suggest_categorical('LGBM_min_child_weight',
                                                                 [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]),
                    subsample = trial.suggest_uniform('LGBM_subsample', 0.2, 0.8),
                    colsample_bytree = trial.suggest_uniform('LGBM_colsample_bytree', 0.4, 0.6),
                    reg_alpha = trial.suggest_categorical('LGBM_alpha',
                                                          [0, 1e-1, 1e-2, 1e-3, 2e-1, 2e-2, 2e-3, 5e-1, 5e-2, 5e-3, 1,
                                                           2, 5, 7, 10, 50, 100]),
                    reg_lambda = trial.suggest_categorical('LGBM_lambda',
                                                           [0, 1e-1, 1e-2, 1e-3, 2e-1, 2e-2, 2e-3, 5e-1, 5e-2, 5e-3, 1,
                                                            5, 10, 20, 50, 100])
                )

                clf = lgb_classifier.fit(
                    X = X_train,
                    y = y_train,
                    eval_metric = trial.suggest_categorical('LGBM_metric', ['multi_logloss']),
                    early_stopping_rounds = 20,
                    eval_set = (X_cross, y_cross),
                    verbose = True

                )

                y_pred = clf.predict(X_cross)
                score = balanced_accuracy_score(y_cross, y_pred)

                final_score += (score / NUMBER_SPLITS)

        return final_score

    def create_submission(self, best_params, submission_path, X_submission, ex_ID, name):

        # if best_params['should_standardise']:
        #     scaler = StandardScaler()
        #     self.X_train = scaler.fit_transform(self.X_train)
        #     X_submission = scaler.transform(X_submission)

        if best_params['standardise'] == 'StandardScaler':
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            X_submission = scaler.transform(X_submission)
        elif best_params['standardise'] == 'L2Norm':
            self.X_train, norms = normalize(self.X_train, norm = 'l2', axis = 0, return_norm = True)
            for index, norm in enumerate(norms):
                X_submission[:, index] = X_submission[:, index] / norm
        elif best_params['standardise'] == 'L1Norm':
            self.X_train, norms = normalize(self.X_train, norm = 'l1', axis = 0, return_norm = True)
            for index, norm in enumerate(norms):
                X_submission[:, index] = X_submission[:, index] / norm
        elif best_params['standardise'] == 'MinMax':
            scaler = MinMaxScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            X_submission = scaler.transform(X_submission)

        if best_params['model_type'] == 'SVC':

            kernel = best_params['SVC_kernel']

            if kernel == 'poly':
                degree = best_params['SVC_degree']
            else:
                degree = 3

            if kernel == 'poly' or kernel == 'sigmoid':
                coef0 = best_params['SVC_coef0']
            else:
                coef0 = 0

            clf = SVC(
                gamma = best_params['SVC_gamma'],
                class_weight = best_params['SVC_class_weight'],
                C = best_params['SVC_C'],
                degree = degree,
                kernel = kernel,
                coef0 = coef0,
                probability = True,
                shrinking = best_params['SVC_shrinking'],
                tol = 0.001,
                verbose = False,
                break_ties = True
            )
            clf.fit(self.X_train, self.y_train)

            predictions = clf.predict(X_submission)

        if best_params['model_type'] == 'LGBM':
            # predictions = best_model.predict(X_submission)
            pass

        ex_test = pd.DataFrame(data = predictions, columns = ['y'])
        export = pd.concat([ex_ID, ex_test], axis = 1)
        export.to_csv(os.path.join(submission_path, f'submission_{name}.csv'), encoding = 'utf-8', index = False)

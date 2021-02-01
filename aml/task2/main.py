#! /usr/bin/python
import argparse
import os

import numpy as np
import optuna
import pandas as pd

from model import TrainingEngine

if __name__ == "__main__":

    np.random.seed(23)

    #################################
    # DEFINE ARGUMENT PARSER
    #################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type = int, default = 1, help = 'Number of threads.')
    parser.add_argument('--n_trials', type = int, default = 1, help = 'Number of trials.')
    parser.add_argument('--name', default = 'svc', help = 'Name of the study.')

    args = parser.parse_args()
    name = args.name

    #################################
    # DEFINE PATHS
    #################################

    base_path = 'data'
    x_train_path = os.path.join(base_path, 'X_train.csv')
    y_train_path = os.path.join(base_path, 'y_train.csv')
    x_test_path = os.path.join(base_path, 'X_test.csv')

    submission_path = 'submission'

    #################################
    # CREATE SUBMISSION PATH
    #################################
    if not os.path.exists(submission_path):
        os.makedirs(submission_path)

    #################################
    # LOAD DATA
    #################################

    # load training data x_train_path
    df_X_train = pd.read_csv(x_train_path, float_precision = "round_trip")
    df_y_train = pd.read_csv(y_train_path, float_precision = "round_trip")

    # load test data dataset
    df_X_test = pd.read_csv(x_test_path, float_precision = "round_trip")
    ex_ID = df_X_test.get('id')  # ids for export

    # create numpy arrays
    X = df_X_train.values[:, 1:]
    y = df_y_train.values[:, 1]

    X_submission = df_X_test.values[:, 1:]

    #################################
    # DEFINE HYPERPARAMETER TUNING
    #################################

    study = optuna.create_study(direction = 'maximize')
    train_engine = TrainingEngine(X, y)
    study.optimize(train_engine, n_trials = args.n_trials,
                   n_jobs = args.n_jobs)

    #################################
    # SAVE STUDY DATAFRAME
    #################################

    study.trials_dataframe().to_csv(os.path.join(submission_path, f'study_{name}.csv'))

    #################################
    # CREATE SUBMISSION
    #################################

    # train_engine.create_submission(study.best_params, submission_path, X_submission,
    #                                ex_ID, name)
    #
    print(study.best_params)

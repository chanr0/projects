#! /usr/bin/python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def aggregate_txt_names():
    txt_file_names = []
    for file in glob.glob("txt_sentoken/neg/*.txt"):
        txt_file_names.append(file)
    for file in glob.glob("txt_sentoken/pos/*.txt"):
        txt_file_names.append(file)
    return txt_file_names


def create_y(reviews):
    y = []
    for review in reviews:
        y.append("pos" in review)
    return y


def create_x(reviews):
    x = []
    for review in reviews:
        file = open(review)
        text = file.read()
        file.close()
        x.append(text)
    return x


if __name__ == "__main__":

    df_X = pd.read_csv("dataset.csv", usecols=["title"])
    df_Y = pd.read_csv("dataset.csv", usecols=["majortopic"])
    X_raw = df_X.values.astype('U').ravel()
    Y = df_Y.values.astype('U').ravel()

    vocab_list = [10**2, 10**3, 10**4, 10**5]
    accuracy_list = []
    Fscore_list = []

    # train with l2 penalty & grid search for hyper parameters
    for vocab in vocab_list:
    
        # TFIDF vectorizer
        stop = stopwords.words("english")
        vectorizer = TfidfVectorizer(max_features=vocab,
                                     stop_words=stop,
                                     ngram_range=(1, 1))
        X = vectorizer.fit_transform(X_raw)
        
        # split into train, test, validation
        X_train, X_dt, y_train, y_dt = train_test_split(X, Y, test_size=0.4)
        X_dev, X_test, y_dev, y_test = train_test_split(X_dt, y_dt, test_size=0.5)
        
        # parameter grid search
        grid = {"C": np.linspace(0.2, 10, 20)}
        LogR = LogisticRegression(penalty='l2',
                                  class_weight="balanced",
                                  multi_class="multinomial",
                                  max_iter=10000)
        clf = GridSearchCV(LogR, grid, cv=5)
        clf.fit(X_dev, y_dev)
        print(clf.best_params_)

        clf = LogisticRegression(C=clf.best_params_["C"],
                                 penalty='l2',
                                 class_weight="balanced",
                                 multi_class="multinomial",
                                 max_iter=10000
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy_list.append(accuracy_score(y_test, y_pred))
        print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))
        Fscore_list.append(f1_score(y_test, y_pred, average='macro'))
    
    # plot accuracy and f1 score as a function of vocabulary size
    plt.figure(0)
    plt.semilogx(vocab_list, accuracy_list, 'bo', label='Accuracy')
    plt.semilogx(vocab_list, Fscore_list, 'go', label='F measure')
    plt.title('Temporal Evolution of Solution')
    plt.xlabel('Vocabulary Size')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('Accuracy_F1_plot')
    plt.show()
#! /usr/bin/python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import glob
from sklearn.feature_extraction.text import TfidfVectorizer


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

    all_reviews = aggregate_txt_names()
    X = TfidfVectorizer().fit_transform(create_x(all_reviews))
    Y = create_y(all_reviews)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))

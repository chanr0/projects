#! /usr/bin/python

import string
import glob
import random
from nltk.tokenize import TreebankWordTokenizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def read_lx(path):
    lex = []
    with open(path) as f:
        for line in f:
            if not line.strip().startswith(';'):
                for word in line.split():
                    lex.append(word)
    f.close()
    return lex


def aggregate_txt_names():
    txt_file_names = []
    for file in glob.glob("txt_sentoken/neg/*.txt"):
        txt_file_names.append(file)
    for file in glob.glob("txt_sentoken/pos/*.txt"):
        txt_file_names.append(file)
    return txt_file_names


def split_test(all_reviews):
    test = []
    # randomly select 400 samples
    for i in range(400):
        rand = random.randint(0, len(all_reviews)-1)
        test.append(all_reviews[rand])
        del all_reviews[rand]
    return all_reviews, test


def compute_f_score(tp, fp, fn):
    precision = tp/(tp + fp)
    recall = tp / (tp + fn)
    return 2*precision*recall / (precision+recall)


def categorize(reviews, pos, neg):
    # true pos, true neg, false pos, false neg
    tp, fp, fn, tn = 0, 0, 0, 0
    for review in reviews:
        file = open(review)
        text = file.read()
        file.close()
        tokens = tokenize(text)
        # check whether there are more positive or negative words
        if len(set(tokens).intersection(set(pos))) - len(set(tokens).intersection(set(neg))) > 0:
            if "pos" in review:
                tp += 1
            else:
                fn += 1
        else:
            if "pos" in review:
                fp += 1
            else:
                tn += 1
    # compute accuracy and F1
    acc = (tp + tn) / len(test_reviews)
    f = compute_f_score(tp, fp, fn)
    return acc, f


def tokenize(text_raw):
    # remove punctuation & stopwords
    stop = set(stopwords.words("english"))
    text_raw = " ".join([word.lower() for word in text_raw.split() if word.lower() not in stop])
    text_raw = text_raw.translate(str.maketrans("", "", string.punctuation))
    # stemming (removes some weird vowels
    text_raw = " ".join([SnowballStemmer("english").stem(word) for word in text_raw.split()])
    # tokenize
    return TreebankWordTokenizer().tokenize(text_raw)


if __name__ == "__main__":
    poslx = read_lx('opinion-lexicon-English/positive-words.txt')
    neglx = read_lx('opinion-lexicon-English/negative-words.txt')
    
    train_reviews, test_reviews = split_test(aggregate_txt_names())

    accuracy, f_score = categorize(test_reviews, poslx, neglx)
    print(accuracy, f_score)

"hw3 classification"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split


def plot(df):
    figure: Figure = plt.figure()

    class_1 = df[df[df.columns[-1]] == 1]
    class_0 = df[df[df.columns[-1]] == 0]

    x = 2
    y = 1

    plt.scatter(class_1[class_1.columns[x]], class_1[class_1.columns[y]], color="red")
    plt.scatter(class_0[class_0.columns[x]], class_0[class_0.columns[y]], color="blue")

    plt.title("banknote dataset")
    figure.savefig("graph.png")


def graph_knn(d):
    figure: Figure = plt.figure()
    colors = ["green", "red", "blue", "purple"]
    for key, c in zip(d, colors):
        plt.scatter([key for item in d[key]], d[key], color=c)
    plt.title("performance of knn")
    plt.ylabel("f1")
    plt.xlabel("k")

    figure.savefig("knn.png")


def graph_logistic():
    figure: Figure = plt.figure()
    colors = ["green", "red", "blue", "purple"]

    d = {
        "c": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "l2": [0.974093, 0.989796, 0.989796, 0.994872, 0.994872, 0.994872, 0.994872],
        "elasticnet": [
            0.822857,
            0.989796,
            0.989796,
            0.994872,
            0.994872,
            0.994872,
            0.994872,
        ],
        "l1": [0.355932, 0.989796, 0.989796, 0.994872, 0.994872, 0.994872, 0.994872],
    }

    plt.plot(d['c'], d['elasticnet'], color='green')
    plt.plot(d['c'], d['l2'], color='blue')
    plt.plot(d['c'], d['l1'], color='red')
    plt.plot(d['c'], [0.4577 for item in d['c']], color='purple')
    plt.plot(d['c'], [0 for item in d['c']], color='purple')
    plt.plot(d['c'], [0.4717 for item in d['c']], color='purple')
    plt.plot(d['c'], [0.6221 for item in d['c']], color='purple')
    plt.plot(d['c'], [1 for item in d['c']], color='orange')

    plt.xscale('log')
    plt.title("performance classifiers")
    plt.ylabel("f1")
    plt.xlabel("c")

    figure.savefig("performance.png")

def clean_float(num):
    "returns a 4 digit float"
    return int(num * 10000) / 1000


def logistic_dev(train_X, train_y, dev_X, dev_y):
    hparams = {
        "newton-cg": ["l2"],
        "lbfgs": ["l2"],
        "liblinear": ["l1", "l2"],
        "sag": ["l2"],
        "saga": ["elasticnet", "l1", "l2"],
    }

    # scoring on the dev set
    results = []
    for item in hparams:
        result = pd.DataFrame(data={"c": [0.001, 0.01, 0.1, 1, 10, 100, 1000]})

        for p in hparams.get(item):
            if p != "elasticnet":
                result[p] = [
                    LogisticRegression(solver=item, penalty=p, C=c)
                    .fit(train_X, train_y)
                    .score(dev_X, dev_y)
                    for c in result["c"]
                ]
            else:
                result[p] = [
                    LogisticRegression(solver=item, penalty=p, C=c, l1_ratio=0.5)
                    .fit(train_X, train_y)
                    .score(dev_X, dev_y)
                    for c in result["c"]
                ]
        results.append(result)

    # f1 scoring on the dev set
    f1 = []
    for item in hparams:
        result = pd.DataFrame(data={"c": [0.001, 0.01, 0.1, 1, 10, 100, 1000]})

        for p in hparams.get(item):
            if p != "elasticnet":
                result[p] = [
                    sklearn.metrics.f1_score(
                        dev_y,
                        LogisticRegression(solver=item, penalty=p, C=c)
                        .fit(train_X, train_y)
                        .predict(dev_X),
                    )
                    for c in result["c"]
                ]
            else:
                result[p] = [
                    sklearn.metrics.f1_score(
                        dev_y,
                        LogisticRegression(solver=item, penalty=p, C=c, l1_ratio=0.5)
                        .fit(train_X, train_y)
                        .predict(dev_X),
                    )
                    for c in result["c"]
                ]
        f1.append(result)

    for result, p in zip(results, hparams):
        print(f"{p}: accuracy")
        print(result)
        print()

    print("\n\n\n")

    for result, p in zip(f1, hparams):
        print(f"{p}: accuracy")
        print(result)
        print()


def baseline_model(train_X, train_y, dev_X, dev_y):
    print("\nbaseline model")
    strategies = ["stratified", "most_frequent", "prior", "uniform", "constant"]
    for strat in strategies:
        dummy = sklearn.dummy.DummyClassifier(strategy=strat, constant=1)
        dummy.fit(train_X, train_y)
        print(f"  dummy strategy = {strat}")

        print(f"  f1: {sklearn.metrics.f1_score(dev_y,dummy.predict(dev_X))}")
        print()


def knn(k, train_X, train_y, dev_X, dev_y):

    y_pred = []

    for x in dev_X.iterrows():
        temp = pd.DataFrame()
        temp["class"] = train_y
        temp["dist"] = ((train_X.sub(x[1]) ** 2).T.sum()) ** 0.5

        neighbors = []

        for _ in range(k):
            nearest = temp[temp["dist"] == temp["dist"].min()]

            n0 = nearest.values[0][0]
            n1 = nearest.values[0][1]

            neighbors.append(int(n0))
            temp = temp[temp["dist"] != temp["dist"].min()]

        # only works accurately with 2 classes
        if sum(neighbors) / len(neighbors) > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    f1 = sklearn.metrics.f1_score(dev_y, y_pred)
    print(f"f1: {f1}")
    return f1


def main():
    df: DataFrame = pd.read_csv("BankNote_Authentication.csv")
    # plot(df)

    # splitting the dataset
    train, test = train_test_split(df, test_size=0.15, shuffle=True)
    train, dev = train_test_split(train, test_size=0.176, shuffle=True)

    # features and targets
    train_X = train[train.columns[:-1]]
    train_y = train[train.columns[-1]]

    # plot(train)

    dev_X = dev[dev.columns[:-1]]
    dev_y = dev[dev.columns[-1]]

    test_X = test[test.columns[:-1]]
    test_y = test[test.columns[-1]]

    ## training and dev

    # clf = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.5).fit(train_X, train_y)
    # print()
    # print('logistic classifier')
    # print(f"f1: {sklearn.metrics.f1_score(dev_y,clf.predict(dev_X))}")
    # print()
    #
    # print('knn')
    # knn(1, train_X, train_y, dev_X, dev_y)

    # logistic_dev(train_X, train_y, dev_X, dev_y)
    # baseline_model(train_X, train_y, dev_X, dev_y)
    # knn(1, train_X, train_y, dev_X, dev_y)

    # graph(d)

    ## evaluation
    clf = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5).fit(
        train_X, train_y
    )
    print()
    print("logistic classifier")
    print(f"f1: {sklearn.metrics.f1_score(test_y,clf.predict(test_X))}")
    print()

    print("knn")
    knn(1, train_X, train_y, test_X, test_y)

    baseline_model(train_X, train_y, test_X, test_y)

    # graph_logistic()

if __name__ == "__main__":
    main()

"""TODO
logistic regression classifier
    default parameters
    evaluate on dev
optimize hyperparameters
    [10000,1000,100,10,1,0.1,0.01,0.001,0.0001]
    evaluate on dev
k nearest neighbors
    evaluate on dev
scikit-learn dummy classifier baseline
    strategy parameters:
        - stratified
        - most frequent
compare all models on test dataset

"""

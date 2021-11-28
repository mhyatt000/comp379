import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yfinance as yf
from matplotlib.figure import Figure
from sklearn import metrics, preprocessing, svm
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

def k_fold_cv(model, df, k: int = 10) -> float:
    """performs k fold cross validation"""

    size = len(df.T.columns)

    folds = []
    sum_score = 0

    for i in range(k - 1):
        df, test = train_test_split(df, test_size=(1 / (k - i)))
        folds.append(test)

    folds.append(df)

    # print(f'len: {len(folds)}')
    # print([len(item.T.columns) for item in folds])

    for i in range(k):
        temp = folds[:i] + folds[i:]
        f = folds[i]

        data = pd.concat(temp)

        model.fit(data[data.columns[:-1]], data["target"])
        score = model.score(f[f.columns[:-1]], f["target"])
        sum_score += score
        # print(f"{i}: {score}")

    # print()
    # print(f"{i}-fold CV: {sum_score/k}")
    return sum_score / k


def grid_search(train_df):
    print('grid_search')

    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    kernel = ["linear", "poly", "rbf", "sigmoid"]
    degree = [1, 2, 3, 4, 5, 6, 7]

    params = {}

    for c in C:
        # for k in kernel:
        #     for d in degree:
        # linear kernel is significantly better

        model = sklearn.linear_model.LogisticRegression(C=c)
        # model = svm.SVC(C=c)

        score = k_fold_cv(model, train_df, 10)
        params[score] = c

        print(f"score: {int(score*10000)/10000}, C={c}")

    print()
    best = max(params.keys())

    return best


def main():
    df = yf.download(tickers="IVV")
    print(df)

    df = df.drop(["High", "Low", "Adj Close", "Volume"], 1)
    df["diff"] = df["Close"] - df["Open"]

    df["diff-1"] = df["diff"].shift(-1)
    df["diff-2"] = df["diff"].shift(-2)
    df["diff-7"] = df["diff"].shift(-7)
    df["diff-30"] = df["diff"].shift(-30)

    df["target"] = df["diff"].apply(lambda x: 1 if x > 0 else 0)

    # df['days'] = [i for i in range(len(df.T.columns))]
    # plot_data(df)

    df = df.drop(["Open", "Close"], 1)
    df.dropna(inplace=True)

    print()

    train, test = train_test_split(df, test_size=0.2)

    diff = list(test["diff"])

    train = train.drop(["diff"], 1)
    test = test.drop(["diff"], 1)

    df = df.drop(["diff"], 1)

    grid_search(train)

    # model = svm.SVC(C=1, kernel='linear', max_iter=10000)

    scores = {}
    dollars = {}

    n=100
    print(f'distribution of scores over {n} iterations')
    for i in range(n):
        train, test = train_test_split(df, test_size=0.2)
        model = sklearn.linear_model.LogisticRegression()
        model.fit(train[train.columns[:-1]], train["target"])
        score = model.score(test[test.columns[:-1]], test["target"])

        score = int(score * 100) / 100

        if score in scores.keys():
            scores[score] += 1
        else:
            scores[score] = 1

        put_or_call = list(model.predict(test[test.columns[:-1]]))
        net = int(sum([val * action for val, action in zip(diff, put_or_call)]))

        if net in dollars.keys():
            dollars[net] += 1
        else:
            dollars[net] = 1

    pprint(scores)

    # plot_accuracy(scores)

    trials_10_000 = {
        0.48: 1,
        0.49: 10,
        0.5: 70,
        0.51: 390,
        0.52: 1178,
        0.53: 2357,
        0.54: 2630,
        0.55: 2148,
        0.56: 927,
        0.57: 241,
        0.58: 37,
        0.59: 10,
        0.6: 1,
    }



def plot_data(df):
    figure: Figure = plt.figure()

    plt.ylabel(ylabel="Stock Value $")
    plt.xlabel(xlabel="Days Since May 19, 2000")
    # plt.xticks(ticks=keys, labels=keys)
    plt.title("Value of IVV")

    a = df[df["target"] == 1]
    b = df[df["target"] == 0]
    plt.plot(df["days"], df["Open"])

    # plt.tight_layout()

    plt.show()
    figure.savefig("value.png")
    figure.clf()
    quit()


def plot_accuracy(data: dict):

    figure: Figure = plt.figure()

    keys = [key for key in data.keys()]
    plt.ylabel(ylabel="Frequency")
    plt.xlabel(xlabel="Accuracy")
    plt.xticks(ticks=keys, labels=keys)
    plt.title("Accuracy of SVM on Stock Market")
    plt.bar(data.keys(), data.values(), width=0.009)
    # plt.tight_layout()

    plt.show()
    figure.savefig("a1.png")
    figure.clf()

def plot_money(data: dict):

    figure: Figure = plt.figure()

    keys = [key for key in data.keys()]
    plt.ylabel(ylabel="Frequency")
    plt.xlabel(xlabel="Net Value $")
    # plt.xticks(ticks=keys, labels=keys)
    plt.title("SVM simulation on Stock Market")
    plt.bar(data.keys(), data.values())
    # plt.tight_layout()

    plt.show()
    figure.savefig("money.png")
    figure.clf()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Warning)
    main()

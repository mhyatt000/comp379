import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn import linear_model

SAMPLES = []


class Entry:
    """
    used for flexible creation of samples (in case fields get added or removed)
    I created this in another project
    also might be useful for testing...
        I can debug with print statements which give the features names
    """

    def __init__(self, name, features, vals):
        """
        initializes obj with a name "type"
        and with a list of dicts
        """

        self.name = name

        self.pairs = {}
        temp = zip(features, vals)
        for item in temp:
            self.pairs[item[0]] = item[1]

    def show(self):
        print(self)
        print(yaml.dump(self.pairs))
        print()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def csv_load(path_csv):
    "returns the entire contents of csv file"
    with open(path_csv, mode="r") as file:
        reader = csv.reader(file, delimiter=",")
        features = next(reader)  # skips first line
        data = []
        for line in reader:
            data.append(line)

    return data, features


def load_objs(csv_name):
    "load csv file into entries"

    data = []
    entries = []
    name = "person"

    "data loading"
    train_path = os.path.dirname(os.path.abspath(__file__)) + "/" + csv_name
    data, features = csv_load(train_path)

    for item in data:
        entries.append(Entry(name, features, item))

    print("loaded")
    return entries


def e(n):
    return 10 ** n


def num(n):
    return float((int(n * 100))) / 100


def activation(n):
    if n > 0.3:
        return 1
    return 0


def main():
    # loading the datasets
    samples_train = load_objs("train.csv")
    samples_test = load_objs("test.csv")
    survival_test = load_objs("gender_submission.csv")

    for sample in samples_test:
        sample.pairs["Survived"] = survival_test[samples_test.index(sample)].pairs.get(
            "Survived"
        )

    # initializing environment
    error_ct = 0
    ct = 0

    X = []
    y = []

    DATASET = samples_test

    # scoring the dataset
    for sample in DATASET:

        score = 0

        x = []
        x.append(sample.pairs.get("Sex"))
        x.append(sample.pairs.get("Age"))
        x.append(sample.pairs.get("SibSp"))
        x.append(sample.pairs.get("Parch"))
        x.append(sample.pairs.get("Pclass"))
        x.append(sample.pairs.get("Fare"))

        if not "" in x:

            if x[0] == "male":
                x[0] = 1
            else:
                x[0] = 0

            x = [float(f) for f in x]

            W = [
                -4.88454530 * e(-1),
                -6.52741481 * e(-3),
                -5.33213771 * e(-2),
                -1.21148713 * e(-2),
                -1.93930826 * e(-1),
                3.07171260 * e(-4),
            ]
            score = np.matmul(x, W)
            score = score

            # multivariable regression
            X.append(x)
            y.append(sample.pairs.get("Survived"))

            # expectations
            if not int(activation(sigmoid(score))) == int(sample.pairs.get("Survived")):
                error_ct += 1
                print(
                    f'predict:{(num(sigmoid(score)))} ... a{(activation(sigmoid(score)))} ... expect: {sample.pairs.get("Survived")}'
                )
            ct += 1

    # error calculation
    print(f"\ngot: {ct-error_ct}/{ct} ... missed: {error_ct}/{ct}")
    print(
        f"accuracy: {int(float(ct-error_ct)*100/ct)}% ... error: {int(float(error_ct)*100/ct)}%"
    )

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    print(regr.coef_)


if __name__ == "__main__":
    main()

"""
    or design a learning rate algorithm
    ∆wj = n(yi - yˆi)xji
    calculate error before backpropogation??
"""

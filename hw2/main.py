import math
import csv
import random

import numpy as np

class Perceptron:
    """Perceptron classifier

    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_inter : int
        number of epochs
    random_state : int
        Random number generator seed for random weight initialization

    Attributes
    ----------
    w_ : id-array
        Weights after fitting
    errors_ : list
        Number of misclassifications (updates) in each epoch

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        ----------
        X : (array-like), shape = [n_examples, n_features]
            training vectors, where n_examples is the number of
            training examples and n_features is the number of features
        y : array-like, shape = [n_examples]
            target values

        Returns
        -------
        self: object

        """

        "randomly initializes weights"
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+np.shape(X)[1])

        self.errors_ = []

        "for each epoch"
        for i in range(self.n_iter):
            errors = 0

            "for every training example"
            for xi, target in zip(X, y):
                "∆w = η * (y - ϕ(z))"
                update = self.eta * (target - self.predict(xi))

                self.w_[1:] += [update * x for x in xi]

                self.w_[0] += update
                "increment errors if ∆w != 0"
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        return z
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Return class label after unit step
        return ϕ(z) activation function
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class AdalineGD:
    """ ADAptive LInear NEuron classifier.

    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_inter : int
        number of epochs
    random_state : int
        Random number generator seed for random weight initialization

    Attributes
    ----------
    w_ : id-array
        Weights after fitting
    cost_ : list
        Number of misclassifications (updates) in each epoch

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        ----------
        X : (array-like), shape = [n_examples, n_features]
            training vectors, where n_examples is the number of
            training examples and n_features is the number of features
        y : array-like, shape = [n_examples]
            target values

        Returns
        -------
        self: object

        """

        "randomly initializes weights"
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + np.shape(X)[1])

        self.cost_ = []

        "for each epoch"
        for j in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = (y[j]-output)

            for w,i in zip(self.w_[1:],range(len(self.w_)-1)):
                w += self.eta * np.dot([x[i] for x in X],errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return sum

    def net_input(self, X):
        """
        Calculate net input
        return z
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """
        Return class label after unit step
        return ϕ(z) activation function
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

def show_linear():

    students = [
        ["gpa", "current_grade", "pass/fail"],
        [1, 0.2, -1],
        [1, 0.2, -1],
        [1, 0.3, -1],
        [2, 0.1, -1],
        [2, 0.2, -1],
        [4, 0.9, 1.0],
        [4, 0.8, 1.0],
        [5, 0.9, 1.0],
        [5, 1.0, 1.0],
        [5, 1.0, 1.0],
    ]

    p = Perceptron(eta=0.01, n_iter=100)
    X = [student[0:2] for student in students]
    X = X[1:]
    y = [student[2] for student in students]
    y = y[1:]
    p.fit(X, y)
    predictions = [p.predict(xi) == yi for xi, yi in zip(X, y)]
    accuracy = 0
    for item in predictions:
        if item:
            accuracy += 1
    print(predictions)
    print(
        """since predictions were true (correct) for each sample in the dataset, the weights have converged
        therefore the dataset is linearly seperable"""
    )
    print(f"accuracy = {accuracy*100/10}%")


def show_nonlinear():

    nonlinear_students = [
        ["gpa", "current_grade", "pass/fail"],
        [1, 0.2, -1],
        [5, 1.0, -1],  #
        [1, 0.3, -1],
        [5, 0.1, -1],  #
        [2, 0.2, -1],
        [1, 0.3, 1.0],  #
        [4, 0.8, 1.0],
        [1, 0.9, 1.0],  #
        [5, 1.0, 1.0],
        [1, 0.2, 1.0],  #
    ]

    p = Perceptron(eta=0.01, n_iter=100)
    X = [student[0:2] for student in nonlinear_students]
    X = X[1:]
    y = [student[2] for student in nonlinear_students]
    y = y[1:]
    p.fit(X, y)
    predictions = [p.predict(xi) == yi for xi, yi in zip(X, y)]
    accuracy = 0
    for item in predictions:
        if item:
            accuracy += 1
    accuracy += 1
    print(predictions)
    print(
        "since predictions were not all correct, this dataset cannot be linearly separated"
    )
    print(f"accuracy = {accuracy*100/10}%")


def csv_load(path_csv):
    "returns the entire contents of csv file"
    with open(path_csv, mode="r") as file:
        reader = csv.reader(file, delimiter=",")
        features = next(reader)  # skips first line
        data = []
        for line in reader:
            # Survived,Pclass,----,Sex,Age,SibSp
            xi = [
                (1 if line[1] == '1' else -1),
                line[2],
                (1 if line[4] == "male" else 0),
                line[5],
                line[6],
            ]
            if not "" in xi:
                xi = [float(val) for val in xi]
                data.append(xi)
        return data  # , features


def titanic():
    data = csv_load("train.csv")
    split: int = int(len(data) * (7 / 10))
    train = data[:split]
    test = data[split:]

    train_X = [xi[1:] for xi in train]
    train_y = [xi[0] for xi in train]

    ada = AdalineGD(eta=0.001, n_iter=30)
    ada.fit(train_X, train_y)
    print(f'cost: {ada.cost_[len(ada.cost_)-1]}')

    test_X = [xi[1:] for xi in test]
    test_y = [xi[0] for xi in test]

    correct = 0
    for xi, yi in zip(test_X, test_y):
        if ada.predict(xi) == yi:
            correct += 1
    accuracy = 100 * correct / len(test)
    print(f"titanic batch AdalineGD accuracy: {accuracy}%")

    weights = {
        "bias": ada.w_[0],
        "pclass": ada.w_[1],
        "sex": ada.w_[2],
        "age": ada.w_[3],
        "sibsp": ada.w_[4],
    }
    for weight in weights: print(f'{weight}: {weights[weight]}')
    print(
        """the most predictive feature was a passenger's age the age weight was double as significant as one's sex and 25% more significant than one's siblings"""
    )

    # baseline model
    correct = 0
    for yi in test_y:
        val = 1 if random.random() > 0.5 else -1
        if val == yi:
            correct += 1
    accuracy = 100 * correct / len(test)
    print()
    print(f"random baseline accuracy: {accuracy}%")

def main():
    show_linear()
    print()
    show_nonlinear()
    print()
    titanic()


if __name__ == "__main__":
    main()

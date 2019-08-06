import numpy as np
from sklearn import linear_model
from sklearn import datasets


class LogisticRegression(object):

    def __init__(self, feture_size):
        self.feture_size = feture_size
        self.weights = np.random.randn(feture_size)
        self.bias = np.random.randn(1)

    def train(self, training_data, test_data,
              etc=1.0,
              epochs=1,
              mini_batch_size=10):

        max_accuracy = 0.0
        for i in range(epochs):
            np.random.shuffle(training_data)
            size = len(training_data)
            mini_batches = [
                training_data[j:j+mini_batch_size]
                for j in range(0, size, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                dw, db = self.update_mini_batch(mini_batch)
                # print(dw)
                # print(db)
                self.weights -= etc * dw
                self.bias -= etc * db

            # print("Epoch {0}: {1}% Accuracy".format(
            #    i + 1, self.evaluate(test_data)))
            acc = self.evaluate(test_data)
            if acc > max_accuracy:
                max_accuracy = acc
            print("Epoch {0}: {1}% Accuracy".format(
                i + 1, acc))
        return max_accuracy
        # print("Maximum Accuracy:{0}".format(max_accuracy))

    def update_mini_batch(self, mini_batch):
        X, Y = zip(*mini_batch)
        X = np.array(X)
        Y = np.array(Y)
        size = len(mini_batch)
        dw, db = np.zeros(self.feture_size), 0
        P = self.predict(X)
        dw = np.dot((P - Y).transpose(), X).transpose()
        db = np.dot((P - Y), np.ones(size).transpose())
        # mod = (np.dot(dw, dw.transpose()) + (db * db)) ** 0.5
        # print(dw, db)

        return (dw / (size), db / (size))

    def predict(self, X):
        return sigmoid(np.dot(X, self.weights.transpose()) + self.bias)

    def cal_loss(self, data):
        X, Y = zip(*data)
        X = np.array(X)
        Y = np.array(Y)
        P = self.predict(X)
        return np.log(np.dot((P ** Y), ((1 - P) ** (1 - Y)))) / -len(data)

    def evaluate(self, data):
        X, Y = zip(*data)
        X = np.array(X)
        Y = np.array(Y)
        P = self.predict(X)
        R = np.multiply((P ** Y), ((1 - P) ** (1 - Y)))
        R = [r for r in R if r > 0.5]
        return len(R) * 100.0 / len(data)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def score(predict_Y):
    pass


def sklearn_test(train_X, train_Y, test_X, test_Y):
    model = linear_model.LogisticRegression()
    model.fit(train_X, train_Y)
    return model.score(test_X, test_Y) * 100


if __name__ == "__main__":
    lr = LogisticRegression(57)
    X = np.loadtxt(
        '/home/goooonite/fanchuiqin/projects/deep-learning-getting-started/datasets/spam_train.csv',
        delimiter=',',
        usecols=(i for i in range(1, 58)))

    Y = np.loadtxt(
        '/home/goooonite/fanchuiqin/projects/deep-learning-getting-started/datasets/spam_train.csv',
        delimiter=',',
        usecols=(58))

    # X, Y = datasets.load_breast_cancer(return_X_y=True)

    training_data = list(zip(X[0:3500], Y[0:3500]))
    test_data = list(zip(X[3500:4000], Y[3500:4000]))
    acc = lr.train(training_data, test_data, epochs=500,
                   etc=1, mini_batch_size=10)

    print(
        "Result of Logistic Regression on Winner or Loser dataset:{0}".format(acc))
    acc = sklearn_test(X[0:3500], Y[0:3500], X[3500:4000], Y[3500:4000])
    print(
        "Result of sklearn Logistic Regression on Winner or Loser dataset:{0}".format(acc))

    X, Y = datasets.load_breast_cancer(return_X_y=True)

    training_data = list(zip(X[0:350], Y[0:350]))
    test_data = list(zip(X[350:], Y[350:]))
    lr = LogisticRegression(30)
    acc = lr.train(training_data, test_data, epochs=500,
                   etc=1, mini_batch_size=10)
    print(
        "Result of Logistic Regression on Breast Cancer dataset:{0}".format(acc))
    acc = sklearn_test(X[0:350], Y[0:350], X[350:], Y[350:])
    print(
        "Result of sklearn Logistic Regression on Breast Cancer dataset:{0}".format(acc))

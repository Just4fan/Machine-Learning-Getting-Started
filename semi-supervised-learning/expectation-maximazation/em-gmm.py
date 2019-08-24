import numpy as np
import matplotlib.pyplot as plot
from math import fabs
from sklearn import datasets, cluster
from collections import Counter


def generate_cluster(mean, conv, sample_size=1):
    return np.random.multivariate_normal(mean, conv, sample_size)


def draw(cluster, label):

    fig = plot.figure()
    colors = ['red', 'green', 'blue']
    axes = fig.add_subplot(111)
    for i in range(len(cluster)):
        x1, x2 = cluster[i]
        axes.scatter(x1, x2, color=colors[int(label[i])])
    plot.show()


class GMM(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def gaussian_distribution(self, X, k):
        conv_det = np.linalg.det(self.conv[k] + np.eye(self.dimension) * 1e-5)
        conv_inv = np.linalg.inv(self.conv[k] + np.eye(self.dimension) * 1e-5)
        # print(conv_det)
        # print(np.power(np.power(2 * np.math.pi, self.dimension) * conv_det, 0.5))

        return (1.0 / np.power(np.power(2 * np.math.pi, self.dimension) * np.fabs(conv_det), 0.5)) \
            * np.exp(-0.5 * np.sum(np.multiply(np.dot((X - self.mean[k]), conv_inv), X - self.mean[k]), axis=1))

    def initialize(self, data):
        # weight_v = np.random.randn(self.n_components)
        self.dimension = data.shape[1]
        # self.weights = [0.5, 0.2, 0.3]
        # self.mean = [[13.2, 10.1], [0.2, 0.8], [5.2, 6.8]]
        # self.conv = [
        #     [[1, 0], [0, 1]],
        #     [[2.1, 0], [0, 2.1]],
        #     [[0.6, 0], [0, 0.6]]
        # ]
        weights = np.random.randn(self.n_components)
        self.weights = np.fabs(weights / np.sum(weights))
        # mean shape(n_components, dimension)
        # mean = np.random.randn(self.dimension * self.n_components)
        # self.mean = np.fabs(np.reshape(
        #     mean, (self.n_components, self.dimension)))
        kmeans = cluster.KMeans(self.n_components)
        kmeans.fit(data)
        self.mean = kmeans.cluster_centers_
        # conv shape(n_components, dimension, dimension)
        conv = np.random.randn(
            self.n_components, self.dimension * self.dimension)
        conv = np.reshape(
            conv, (self.n_components, self.dimension, self.dimension))
        for k in range(self.n_components):
            conv[k] = np.triu(conv[k])
            conv[k] = conv[k] + (conv[k].transpose() -
                                 np.diag(conv[k].diagonal()))
        self.conv = conv
        # print(self.weights)
        # print(self.mean)
        # print(self.conv)

    def confusion_matrix(self, predict, label):
        pass

    def evaluate(self, test_x, test_y):
        size = len(test_x)
        y_matrix = [self.gaussian_distribution(test_x, k)
                    for k in range(self.n_components)]
        predict_y = np.argmax(y_matrix, axis=0)

    def score(self, test_x, test_y):
        size = len(test_x)
        y_matrix = [self.gaussian_distribution(test_x, k)
                    for k in range(self.n_components)]
        # for k in range(self.n_components):
        #     print(y_matrix[k])
        py = []

        for i in range(size):
            max = 0
            t = -1
            for k in range(self.n_components):
                if y_matrix[k][i] > max:
                    max = y_matrix[k][i]
                    t = k
            py.append(t)

        err = 0
        for k in range(self.n_components):
            index = [y for y in range(len(test_y)) if int(test_y[y]) == k]
            # print(index)
            tp = [py[i] for i in index]
            print(tp)
            max = -1
            c = Counter(np.array(tp))
            print(c)
            for k in range(self.n_components):
                # print(c[k])
                if max < c[k]:
                    max = c[k]
            err += len(tp) - max
        print(test_y)
        print(py)

        return err, py

    def likelihood(self, data):
        y_matrix = np.array([self.gaussian_distribution(data, k) * self.weights[k]
                             for k in range(self.n_components)])
        log_y = -np.log(np.sum(y_matrix, axis=0))
        # print(log_y.shape)
        print(np.sum(log_y))
        return np.sum(log_y)

    def fit(self, data):
        size = len(data)
        # data shape(size, dimension)
        self.initialize(data)
        ll = 0
        ll_c = self.likelihood(data)
        while fabs(ll - ll_c) > 1e-5:
            ll = ll_c
            # y_martix shape(n_components, size)
            # E-step
            y_matrix = np.array([self.gaussian_distribution(data, k) * self.weights[k]
                                 for k in range(self.n_components)])
            y_matrix = y_matrix / np.sum(y_matrix, axis=0)

            # M-step
            t_mean = self.mean
            self.mean = (np.dot(y_matrix, data).transpose() /
                         np.sum(y_matrix, axis=1)).transpose()
            for k in range(self.n_components):
                e = data - t_mean[k]
                self.conv[k] = np.dot(
                    y_matrix[k] * e.transpose(), e) / np.sum(y_matrix[k])
            self.weights = np.sum(y_matrix, axis=1) / size
            ll_c = self.likelihood(data)
            # print(ll_c)
            # print(self.mean)
            # print(self.conv)
            # print(self.weights)


if __name__ == "__main__":
    size = 300
    mean = [[13.2, 5.1], [0.2, 0.8], [5.2, 6.8]]
    conv = [[[1, 0], [0, 1]], [[2.1, 0], [0, 2.1]], [[0.6, 0], [0, 0.6]]]
    data = np.random.choice([0, 1, 2], size=size,
                            replace=True, p=[0.5, 0.2, 0.3])
    # print(data)
    X, Y = datasets.load_iris(return_X_y=True)
    X = np.zeros(shape=(size, 2))
    Y = np.zeros(shape=(size,))
    for i in range(len(data)):
        X[i] = np.random.multivariate_normal(mean[data[i]], conv[data[i]])
        Y[i] = data[i]
    gmm = GMM(3)
    # gmm.initialize(X, 3)
    gmm.fit(X)
    err, py = gmm.score(X, Y)
    print("ERR: {0} / {1}".format(err, len(X)))
    # print("Accuracy: {0} / {1}".format(result, size))
    print(gmm.mean)
    print(mean)
    print(gmm.conv)
    print(conv)
    print(gmm.weights)
    draw(X, py)

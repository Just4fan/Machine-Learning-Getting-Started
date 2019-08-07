import numpy as np


# 生成随机数据，pa，pb，pc分别是A, B, C三枚硬币正面的概率
def generate_data(pa, pb, pc, size):
    data = np.random.choice([0, 1], size=size, replace=True, p=[1 - pa, pa])
    for i in range(size):
        p = (pb ** data[i]) * (pc ** data[i])
        data[i] = np.random.choice([0, 1], p=[1 - p, p])

    return data


def EM(w, data):
    tw = np.zeros(3)
    while np.dot(w - tw, w - tw) > 0.1:
        # print("w: {0}".format(w))
        # print("tw: {0}".format(tw))
        tw = np.array(w)
        n = len(data)
        mu = np.zeros(n)
        # print("w: {0}".format(w))
        for i in range(n):
            d1 = w[0] * (w[1] ** data[i]) * ((1 - w[1]) ** (1 - data[i]))
            d2 = (1 - w[0]) * (w[2] ** data[i]) * ((1 - w[2]) ** (1 - data[i]))
            mu[i] = d1 / (d1 + d2)
            # print("d1: {0}, d2: {1}".format(d1, d2))
            # print("mu {0} : {1}".format(i, mu[i]))
        w[0] = np.dot(mu, np.ones(n)) / n
        w[1] = np.dot(mu, data) / np.dot(mu, np.ones(n))
        w[2] = np.dot(1 - mu, data) / np.dot(1 - mu, np.ones(n))
        # print(w)

    return w


if __name__ == "__main__":
    data = generate_data(0.7, 0.4, 0.6, 10)
    # data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    # data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    # print(data)
    w = np.array([0.4, 0.6, 0.7])
    w = EM(w, data)
    print(w)

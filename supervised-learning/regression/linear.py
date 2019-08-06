import numpy as np
import pandas as pd
from sklearn import linear_model
# from sklearn import datasets


def process_data(data):
    train_x, train_y = np.zeros(shape=(0, 17)), np.zeros(shape=(0,))
    data = data.replace(['NR'], [0.0])
    data = np.array(data).astype(float)
    for i in range(0, len(data), 18):
        temp_x = np.append(data[i:i+9], data[i+10:i+18], axis=0)
        temp_y = data[i+9]
        temp_x = temp_x.T
        # print(temp_x)
        train_x = np.append(train_x, temp_x, axis=0)
        train_y = np.append(train_y, temp_y, axis=0)

    return train_x, train_y


def train(training_data,
          test_data=None,
          mini_batch_size=20,
          etc=1,
          epochs=1):
    weights = np.random.randn(17)
    bias = np.random.randn(1)
    min_loss = np.inf

    for i in range(epochs):
        np.random.shuffle(training_data)
        size = len(training_data)
        mini_batches = [
            training_data[j:j+mini_batch_size]
            for j in range(0, size, mini_batch_size)
        ]

        for mini_batch in mini_batches:
            dw, db = update_mini_batch(mini_batch, weights, bias)
            weights -= etc * dw
            bias -= etc * db
            loss = evaluate(mini_batch, weights, bias)
            print("batch loss {0}".format(loss))

        if test_data:
            loss = evaluate(test_data, weights, bias)
            if loss < min_loss:
                min_loss = loss
            # print("Epoch {0}: test data loss {1}".format(i, loss))

    # print(weights)
    return min_loss


def update_mini_batch(mini_batch, w, b):
    dw, db = np.zeros(17), 0
    for x, y in mini_batch:
        loss = cal_loss(x, y, w, b)
        dw -= loss * x
        db -= loss

    mod = (np.dot(dw, dw.T) + db ** 2) ** 0.5
    return (dw / (mod * len(mini_batch)), db / (mod * len(mini_batch)))


def evaluate(test_data, w, b):
    test_x, test_y = zip(*test_data)
    # loss = test_y - (test_x * w + b)
    loss = test_y - (np.dot(test_x, w.T) + b)
    return np.dot(loss, loss.T) / len(test_x)


def cal_loss(x, y, w, b):
    return y - (np.dot(x, w.T) + b)


def sklearn(train_x, train_y, test_x, test_y):
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)
    loss = test_y - model.predict(test_x)
    loss = np.dot(loss, loss.T)
    print(loss / len(test_x))


data = pd.read_csv(
    '/home/goooonite/fanchuiqin/projects/deep-learning-getting-started/datasets/pm_data.csv',
    usecols=[i for i in range(3, 27)])
train_x, train_y = process_data(data)
data = zip(train_x, train_y)
data = list(data)
loss = train(data[0:4000], data[4000:], etc=1,
             epochs=500, mini_batch_size=4000)
print("Minimum Loss: {0}".format(loss))
sklearn(train_x[0:4000], train_y[0:4000], train_x[4000:], train_y[4000:])


# boston dataset 参数小幅度变化会导致结果发生较大改变，容易错过最优解
# 需要降低学习率，增加迭代次数

# train_x, train_y = datasets.load_boston(return_X_y=True)
# data = list(zip(train_x, train_y))
# loss = train(data[0:400], data[400:], etc=0.01,
#              epochs=20000, mini_batch_size=400)
# print(loss)
# sklearn(train_x[0:400], train_y[0:400], train_x[400:], train_y[400:])

# CRIM--城镇人均犯罪率
# ZN - 占地面积超过25,000平方英尺的住宅用地比例。
# INDUS - 每个城镇非零售业务的比例。
# CHAS - Charles River虚拟变量（如果是河道，则为1;否则为0）
# NOX - 一氧化氮浓度（每千万份）
# RM - 每间住宅的平均房间数
# AGE - 1940年以前建造的自住单位比例
# DIS加权距离波士顿的五个就业中心
# RAD - 径向高速公路的可达性指数
# TAX  - 每10,000美元的全额物业税率
# PTRATIO - 城镇的学生与教师比例
# B - 1000（Bk - 0.63）^ 2其中Bk是城镇黑人的比例
# LSTAT - 人口状况下降％
# MEDV - 自有住房的中位数价值1000美元

import numpy as np

file = open('housing_scale.txt', 'r')
label = np.zeros((506, 1), dtype=np.float32)
data = np.zeros((506, 13), dtype=np.float32)
for i, line in enumerate(file):
    line = line.strip('\n').split()
    # print(i, line, float(line[0]))
    label[i] = float(line[0])
    for j in range(13):
        data[i, j] = float(line[j+1].split(':')[1])
        # print(float(line[j+1].split(':')[1]))

w = np.random.randn(13, 1)/100.
b = np.zeros((1, 1), dtype=np.float32)
# data_b = np.concatenate((data, np.ones((506, 1))), axis=1)
# w_b = np.concatenate((w, b), axis=0)
# print(data_b.shape, w_b.shape)

loss_plot = []
iter_plot = []
lr = 0.0001
for i in range(1000):
    yp = np.dot(data, w) + b
    # print(yp.shape)
    err = label - yp
    w = w + lr * (np.dot(err.transpose(), data)).transpose()
    b = b + lr * np.sum(err)
    loss = 1/2 * np.mean(np.square(err))
    loss_plot.append(loss)
    iter_plot.append(i)

import matplotlib.pyplot as plt
plt.plot(iter_plot, loss_plot, color='green', label='training_loss')
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()





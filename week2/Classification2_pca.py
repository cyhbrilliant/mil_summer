import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def Dsigmoid(x):
    return x*(1-x)

def cross_entropy_loss(label, yp):
    return -np.sum(label*np.log(yp) + (1-label)*np.log(1-yp))

file = open('data.csv', 'r')
label = np.zeros((569, 1), dtype=np.float32)
data = np.zeros((569, 30), dtype=np.float32)
for i, line in enumerate(file):
    line = line.strip('\n').split(',')
    # print(i, line)
    label[i] = 1 if line[1] == 'M' else 0
    for j in range(30):
        data[i, j] = float(line[j+2])

# data_min = np.min(data, axis=1)[:,np.newaxis]
# data_max = np.max(data, axis=1)[:,np.newaxis]
# data = (data - data_min)/(data_max - data_min)
data -= np.mean(data, 1)[:,np.newaxis]
data /= np.std(data, 1)[:,np.newaxis]
#pca
data_t = data.transpose()
data_cov = np.cov(data_t)
evals , evecs = np.linalg.eig(data_cov)
data = np.dot(data, evecs)[:, :15]
# a = np.dot(x, evecs)
# print(evecs.shape)
# print(evals)

# print(data)


trainimg = data[0:400]
valimg = data[400:569]
trainlable = label[0:400]
vallable = label[400:569]

# W1=np.random.randn(10,50)/10000.
# W2=np.random.randn(50,1)/10000.
W1=np.zeros([15,50])
W2=np.zeros([50,1])
B1=np.zeros([1,50])
B2=np.zeros([1,1])



lr = 0.001
loss_a = 0
train_plot = []
val_plot = []
acc_plot = []
iter_plot = []
for i in range(30000):
    L1OUT = sigmoid(np.dot(trainimg, W1) + B1)
    L2OUT = sigmoid(np.dot(L1OUT, W2) + B2)

    ERR = trainlable - L2OUT
    W2 = W2 + lr * (np.dot(ERR.transpose(), L1OUT)).transpose()
    B2 = B2 + lr * np.sum(ERR)
    W1 = W1 + lr * (np.dot((np.dot(ERR, W2.transpose())*Dsigmoid(L1OUT)).transpose(), trainimg)).transpose()
    B1 = B1 + lr * np.sum(np.dot(ERR, W2.transpose())*Dsigmoid(L1OUT), axis=0)

    loss = cross_entropy_loss(trainlable, L2OUT)
    loss_a += loss
    # print(loss)

    if (i+1)%100 == 0:
        L1OUT_t = sigmoid(np.dot(valimg, W1) + B1)
        L2OUT_t = sigmoid(np.dot(L1OUT_t, W2) + B2)
        loss_val = cross_entropy_loss(vallable, L2OUT_t)
        L2OUT_t[L2OUT_t>0.5] = 1
        L2OUT_t[L2OUT_t<=0.5] = 0
        acc = np.mean((L2OUT_t == vallable).astype(np.float32))
        print('acc = ', acc, ', loss = ', loss_a/300., ', loss_val = ', loss_val)
        acc_plot.append(acc*100.)
        train_plot.append(loss_a/300.)
        val_plot.append(loss_val)
        iter_plot.append(i)
        loss_a = 0

import matplotlib.pyplot as plt
plt.plot(iter_plot, train_plot, color='green', label='training_loss')
plt.plot(iter_plot, val_plot, color='red', label='val_loss')
plt.plot(iter_plot, acc_plot, color='blue', label='val_accuracy')
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()

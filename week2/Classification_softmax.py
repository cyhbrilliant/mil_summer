import numpy as np
import Perceptron.read_minst as read

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def Dsigmoid(x):
    return x*(1-x)

def Lrelu(x):
    x[x<0] *= 0.1
    x = np.clip(x, -1, 1)
    return x

def DLrelu(x):
    g = np.zeros_like(x)
    g += 0.1
    g[x>0] = 1
    return g

def softmax(x):
    # print(np.sum(x))
    x_exp = np.exp(x)
    sum = np.sum(x_exp, axis=1)
    return x_exp/sum[:, np.newaxis]

def cross_entropy_loss(label, yp):
    yp = np.clip(yp, 0.0000001, 1000000)
    return -np.sum(label*np.log(yp))

trainimg = read.getTrainimg()
trainlable = read.getTrainlabel()
valimg = read.getValimg()
vallable = read.getVallabel()
trainlable_onehot = []
vallable_onehot = []
for i in range(trainlable.shape[0]):
    temp = np.zeros((10))
    temp[trainlable[i]] = 1
    # print(temp)
    trainlable_onehot.append(temp)
trainlable_onehot = np.array(trainlable_onehot)
print(trainlable_onehot.shape)
for i in range(vallable.shape[0]):
    temp = np.zeros((10))
    temp[vallable[i]] = 1
    # print(temp)
    vallable_onehot.append(temp)
vallable_onehot = np.array(vallable_onehot)
print(vallable_onehot.shape)

W1=np.random.randn(784,512)/100.
W2=np.random.randn(512,10)/100.
B1=np.zeros([1,512])
B2=np.zeros([1,10])

batch_size = 50
def get_data():
    Databatch = []
    Labelbatch = []
    for i in range(batch_size):
        index = np.random.randint(0, trainimg.shape[0])
        Labelbatch.append(trainlable_onehot[index])
        Databatch.append(trainimg[index])
    return np.array(Databatch), np.array(Labelbatch)

lr = 0.01
loss_a = 0
train_plot = []
val_plot = []
acc_plot = []
iter_plot = []
for i in range(10000):
    data, label = get_data()
    L1OUT = sigmoid(np.dot(data, W1) + B1)
    L2OUT = softmax(np.dot(L1OUT, W2) + B2)

    ERR = label - L2OUT
    W2 = W2 + lr * (np.dot(ERR.transpose(), L1OUT)).transpose()
    B2 = B2 + lr * np.sum(ERR)
    W1 = W1 + lr * (np.dot((np.dot(ERR, W2.transpose())*Dsigmoid(L1OUT)).transpose(), data)).transpose()
    B1 = B1 + lr * np.sum(np.dot(ERR, W2.transpose())*Dsigmoid(L1OUT), axis=0)

    L1OUT = Lrelu(np.dot(data, W1) + B1)
    L2OUT = softmax(np.dot(L1OUT, W2) + B2)
    #
    # ERR = label - L2OUT
    # W2 = W2 + lr * (np.dot(ERR.transpose(), L1OUT)).transpose()
    # B2 = B2 + lr * np.sum(ERR)
    # W1 = W1 + lr * (np.dot((np.dot(ERR, W2.transpose())*DLrelu(L1OUT)).transpose(), data)).transpose()
    # B1 = B1 + lr * np.sum(np.dot(ERR, W2.transpose())*DLrelu(L1OUT), axis=0)

    # L1OUT = np.dot(data, W1) + B1
    # L2OUT = softmax(np.dot(L1OUT, W2) + B2)
    #
    # ERR = label - L2OUT
    # W2 = W2 + lr * (np.dot(ERR.transpose(), L1OUT)).transpose()
    # B2 = B2 + lr * np.sum(ERR)
    # W1 = W1 + lr * (np.dot((np.dot(ERR, W2.transpose())).transpose(), data)).transpose()
    # B1 = B1 + lr * np.sum(np.dot(ERR, W2.transpose()), axis=0)

    loss = cross_entropy_loss(label, L2OUT)
    loss_a += loss
    # print(loss)

    if (i+1)%100 == 0:
        L1OUT_t = sigmoid(np.dot(valimg, W1) + B1)
        # L1OUT_t = Lrelu(np.dot(valimg, W1) + B1)
        # L1OUT_t = np.dot(valimg, W1) + B1
        L2OUT_t = softmax(np.dot(L1OUT_t, W2) + B2)
        loss_val = cross_entropy_loss(vallable_onehot, L2OUT_t)
        L2OUT_arg = np.argmax(L2OUT_t, axis=1)
        acc = np.mean((L2OUT_arg == vallable).astype(np.float32))
        print('acc = ', acc, ', loss = ', loss_a/100., ', loss_val = ', loss_val/(10000./batch_size))
        acc_plot.append(acc*100.)
        train_plot.append(loss_a/100.)
        val_plot.append(loss_val/(10000./batch_size))
        iter_plot.append(i)
        loss_a = 0

import matplotlib.pyplot as plt
plt.plot(iter_plot, train_plot, color='green', label='training_loss')
plt.plot(iter_plot, val_plot, color='red', label='val_loss')
plt.plot(iter_plot, acc_plot, color='blue', label='val_accuracy')
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()

import numpy as np
import Perceptron.read_minst as read

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def Dsigmoid(x):
    return x*(1-x)

trainimg = read.getTrainimg()
trainlable = read.getTrainlabel()[:, np.newaxis]
valimg = read.getValimg()
vallable = read.getVallabel()[:, np.newaxis]

W1=np.random.randn(784,512)/100.
W2=np.random.randn(512,1)/100.
B1=np.zeros([1,512])
B2=np.zeros([1,1])

batch_size = 40
def get_data():
    Databatch = []
    Labelbatch = []
    for i in range(batch_size):
        index = np.random.randint(0, trainimg.shape[0])
        Labelbatch.append(trainlable[index])
        Databatch.append(trainimg[index])
    return np.array(Databatch), np.array(Labelbatch)

lr = 0.0001
for i in range(100000):
    data, label = get_data()
    L1OUT = sigmoid(np.dot(data, W1) + B1)
    L2OUT = np.dot(L1OUT, W2) + B2

    ERR = label - L2OUT
    W2 = W2 + lr * (np.dot(ERR.transpose(), L1OUT)).transpose()
    B2 = B2 + lr * np.sum(ERR)
    W1 = W1 + lr * (np.dot((np.dot(ERR, W2.transpose())*Dsigmoid(L1OUT)).transpose(), data)).transpose()
    B1 = B1 + lr * np.sum(np.dot(ERR, W2.transpose())*Dsigmoid(L1OUT), axis=0)

    loss = 1 / 2 * np.mean(np.square(ERR))
    print(loss)

while True:
    a=input("身高：")
    b=input("体重：")
    # c=input("san:")1
    testData=np.array([[a,b]],dtype=float)
    L1OUT = sigmoid(np.dot(testData, W1) + B1)
    L2OUT = np.dot(L1OUT, W2) + B2
    print(L2OUT)

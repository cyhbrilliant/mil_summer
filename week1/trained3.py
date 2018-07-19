import time
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import h5py
import os
import Knn.read_minst as read
#0.83

IsTraining=False
IsLoad_state_dict=False
IsGpuEval=True

if IsTraining:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
else:
    if IsGpuEval:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def ModelCheck(net):
    params = list(net.parameters())
    print(len(params))
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("参数和：" + str(l))
        k = k + l
    print("总参数和：" + str(k))
    print(len(list(net.children())))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = torch.nn.Dropout2d(0.3)

        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=128)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = torch.nn.Dropout2d(0.3)

        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(num_features=256)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=7)

        self.dropout3 = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(256, 10)

    def forward(self, x):
        print(x.size())
        x = F.relu(self.conv1(x))
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv2(x)))))
        x = F.relu(self.conv3(x))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv4(x)))))
        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.bn3(self.conv6(x)))).view([-1, 256])
        feature = x
        x = self.dropout3(x)
        x = self.fc1(x)
        return x, feature
net = Net()

trainimg = read.getTrainimg()
trainimg_pic = np.reshape(trainimg, [-1, 1, 28, 28])
trainlable = read.getTrainlabel()
valimg = read.getValimg()
valimg_pic = np.reshape(valimg, [-1, 1, 28, 28])
vallable = read.getVallabel()

if IsTraining:
    if IsLoad_state_dict:
        net.load_state_dict(torch.load('./3.pkl')) #载入参数

    net.train()
    net.cuda()
    Loss_fn = torch.nn.CrossEntropyLoss()
    Optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    def getBatch(batchsize):
        Databatch = []
        Labelbatch = []
        for i in range(batchsize):
            index = np.random.randint(0, trainimg_pic.shape[0])
            Labelbatch.append(trainlable[index])
            Databatch.append(trainimg_pic[index])
        return np.array(Databatch), np.array(Labelbatch)

    #hyper_param
    batchsize=12
    for iter in range(5000000):
        Data_batch,Label_batch=getBatch(batchsize=batchsize)
        Input=torch.autograd.Variable(torch.from_numpy(Data_batch.astype(np.float32))).cuda()
        Label=torch.autograd.Variable(torch.from_numpy(Label_batch.astype(np.int64))).cuda()
        Predic, Feature = net(Input)
        loss=Loss_fn(Predic,Label)
        print(iter,loss.cpu().data.numpy())
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

        if (iter+1)%1000==0:
            torch.save(net.state_dict(), './3.pkl')
            print('Save Model')

else:
    if IsGpuEval:
        net.load_state_dict(torch.load('./3.pkl'))  # 载入参数
        net.eval()
        net.cuda()
    else:
        net.load_state_dict(torch.load('./3.pkl', map_location=lambda storage, loc: storage))  # 载入参数
        net.eval()

    def getBatchPic(startnum,endnum):
        Data = valimg_pic[startnum:endnum]
        Label = vallable[startnum:endnum]
        return Data,Label

    Data_batch,Label_batch=getBatchPic(0, 5000)
    Input = torch.autograd.Variable(torch.from_numpy(Data_batch.astype(np.float32)))
    if IsGpuEval:
        Input = Input.cuda()
    Predict, Feature = net(Input)
    if IsGpuEval:
        Predict = Predict.cpu().data.numpy()
        Feature = Feature.cpu().data.numpy()
    else:
        Predict = Predict.data.numpy()
        Feature = Feature.data.numpy()

    class_result = np.argmax(Predict, axis=1)
    print('accuracy = ', np.mean((class_result==Label_batch).astype(np.float32)))

    #
    #
    # k = 20
    # valnum = 100
    # val = Feature
    # vall = vallable[0: valnum]
    # train = np.zeros((60000, 256))
    # trainl = trainlable
    # for i in range(60):
    #     Data_batch = trainimg_pic[i*1000:(i+1)*1000]
    #     Input = torch.autograd.Variable(torch.from_numpy(Data_batch.astype(np.float32)))
    #     if IsGpuEval:
    #         Input = Input.cuda()
    #     l, p = net(Input)
    #     if IsGpuEval:
    #         p = p.cpu().data.numpy()
    #     else:
    #         p = p.data.numpy()
    #     train[i*1000:(i+1)*1000] = p
    #
    #
    # def Euclidean(a, b):
    #     return np.sqrt(np.sum(np.square(a - b)))
    #
    # def Manhattan(a, b):
    #     return np.sum(np.abs(a - b))
    #
    # def k_vote(kvec, k):
    #     vote = np.zeros((k))
    #     for iteri in range(k):
    #         vnum = 0
    #         for iterj in range(k):
    #             if trainl[kvec[iteri]] == trainl[kvec[iterj]]:
    #                 vnum += 1
    #         vote[iteri] = vnum
    #     # print(kvec)
    #     # print(np.argmax(vote))
    #     return trainl[kvec[np.argmax(vote)]]
    #
    #
    # dist_result = np.zeros((valnum, 60000))
    # for i in range(valnum):
    #     for j in range(60000):
    #         dist_result[i, j] = Euclidean(val[i], train[j])
    #     print(i)
    #
    # class_result = np.zeros((valnum), np.int32)
    # for i in range(valnum):
    #     sort = np.argsort(dist_result[i])
    #     class_result[i] = k_vote(sort[0:k], k)
    #     # print(class_result[i])
    #     # print(dist_result[i, sort[0]])
    #     # print(dist_result[i, sort[1]])
    #     # print(dist_result[i, sort[2]])
    #     # print(dist_result[i, sort[3]])
    #
    # print('accuracy = ', np.mean((class_result == vall).astype(np.float32)))

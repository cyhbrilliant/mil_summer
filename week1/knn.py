import Knn.read_minst as read
import numpy as np

print(read.getTrainimg().shape)
print(read.getTrainlabel().shape)
print(read.getValimg().shape)
print(read.getVallabel().shape)

trainimg = read.getTrainimg()
trainlable = read.getTrainlabel()
valimg = read.getValimg()
vallable = read.getVallabel()

k = 20
trainnum = 200
valnum = 100
trainimg = trainimg[0: trainnum]
trainlable = trainlable[0: trainnum]
valimg = valimg[0: valnum]
vallable = vallable[0: valnum]

def Euclidean(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

def  Manhattan(a, b):
    return np.sum(np.abs(a-b))

def k_vote(kvec, k):
    vote = np.zeros((k))
    for iteri in range(k):
        vnum = 0
        for iterj in range(k):
            if trainlable[kvec[iteri]] == trainlable[kvec[iterj]]:
                vnum += 1
        vote[iteri] = vnum
    # print(kvec)
    # print(np.argmax(vote))
    return trainlable[kvec[np.argmax(vote)]]


dist_result = np.zeros((valnum, trainnum))
for i in range(valnum):
    for j in range(trainnum):
        dist_result[i, j] = Euclidean(valimg[i], trainimg[j])
    print(i)

class_result = np.zeros((valnum), np.int32)
for i in range(valnum):
    sort = np.argsort(dist_result[i])
    class_result[i] = k_vote(sort[0:k], k)
    # print(class_result[i])
    # print(dist_result[i, sort[0]])
    # print(dist_result[i, sort[1]])
    # print(dist_result[i, sort[2]])
    # print(dist_result[i, sort[3]])

print('accuracy = ', np.mean((class_result==vallable).astype(np.float32)))

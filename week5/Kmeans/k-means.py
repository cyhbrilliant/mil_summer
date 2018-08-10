import numpy as np
import matplotlib.pyplot as plt
from Kmeans.read_minst import DataLoader


class KMeans():
    def __init__(self, K=10, dist_mode = 'Euclidean', iteration=10):
        # load hyper param
        self.K = K
        self.dist_mode = dist_mode
        self.iteration = iteration

        # load data
        '''
            self.trainimg.shape -> (60000, 784)
            self.trainlable.shape -> (60000)
            self.valimg.shape -> (10000, 784)
            self.vallable.shape -> (10000)
        '''
        dataloader = DataLoader()
        self.trainimg = dataloader.getTrainimg()
        self.trainlable = dataloader.getTrainlabel()
        self.valimg = dataloader.getValimg()
        self.vallable = dataloader.getVallabel()

        # data stucture
        '''
            self.attr.shape = (60000)
            self.cluster.shape = (10, 784)
        '''
        self.attr = np.zeros_like(self.trainlable)
        random_trainimg = self.trainimg.copy()
        np.random.shuffle(random_trainimg)
        self.cluster = random_trainimg[:self.K] # random choose cluster without repeat

        # draw curve var
        self.curve_ix = []
        self.curve_loss = []
        self.curve_acct = []
        self.curve_accv = []

    def Euclidean(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def Manhattan(self, a, b):
        return np.sum(np.abs(a - b))

    def compute(self):
        '''
            dist.shape -> (60000, 10)
            attr.shape -> (60000)
        '''
        for ix in range(self.iteration):
            dist = np.array([[self.Euclidean(cl, sample) if self.dist_mode == 'Euclidean' else self.Manhattan(cl, sample) for cl in self.cluster] for sample in self.trainimg])
            self.attr = np.argmin(dist, axis=1)

            cluster_ = np.zeros((self.K, self.trainimg.shape[1]))
            cluster_num = np.zeros(self.K)
            for sample_ix, sample_attr in enumerate(self.attr):
                cluster_[sample_attr] += self.trainimg[sample_ix]
                cluster_num[sample_attr] += 1
            cluster_ /= cluster_num[:, np.newaxis]

            loss = self.Manhattan(self.cluster,cluster_)
            self.cluster = cluster_

            acc_train = self.acc('train')
            acc_val = self.acc('val')
            self.curve_ix.append(ix)
            self.curve_loss.append(loss)
            self.curve_acct.append(acc_train)
            self.curve_accv.append(acc_val)
            print(ix, loss, acc_train, acc_val)
            # self.draw_distribution()

    def acc(self, mode = 'train'):
        attr = self.attr if mode == 'train' else np.argmin(np.array([[self.Euclidean(cl, sample) if self.dist_mode == 'Euclidean' else self.Manhattan(cl, sample)
                          for cl in self.cluster] for sample in self.valimg]), axis=1)
        transf = np.zeros((10, 10))
        for attr_i, attr_v in enumerate(attr):
            if mode == 'train':
                transf[attr_v, self.trainlable[attr_i]] += 1
            else:
                transf[attr_v, self.vallable[attr_i]] += 1
        transf = np.argmax(transf, axis=1)

        if mode == 'train':
            return np.mean((np.array([transf[attr[ix]] for ix in range(attr.__len__())]) == self.trainlable).astype(np.float32))
        else:
            return np.mean((np.array([transf[attr[ix]] for ix in range(attr.__len__())]) == self.vallable).astype(np.float32))

    def draw_curve(self):
        plt.plot(self.curve_ix, self.curve_loss, color='green', label='curve_loss')
        plt.xlabel('iteration times')
        plt.ylabel('loss')
        plt.show()
        plt.plot(self.curve_ix, self.curve_acct, color='red', label='curve_acct')
        plt.plot(self.curve_ix, self.curve_accv, color='blue', label='curve_accv')
        plt.xlabel('iteration times')
        plt.ylabel('rate')
        plt.show()

    def PCA(self, data):
        '''
            data.shape -> (n, 784)
            output.shape -> (n, 2)
        '''
        data -= np.mean(data, 1)[:, np.newaxis]
        data /= np.std(data, 1)[:, np.newaxis]
        data_t = data.transpose()
        data_cov = np.cov(data_t)
        evals, evecs = np.linalg.eig(data_cov)
        return np.dot(data, evecs[:, :2])

    def TSNE(self, data):
        from sklearn.manifold import TSNE
        tsne = TSNE()
        data = tsne.fit_transform(data)
        return data

    def draw_distribution(self):
        coordi_train = self.TSNE(np.concatenate((self.trainimg[:1000], np.array(self.cluster)), axis=0))
        # coordi_train = self.PCA(np.concatenate((self.trainimg[:1000], np.array(self.cluster)), axis=0))

        coordi_train_cl_x = coordi_train[1000:, 0]
        coordi_train_cl_y = coordi_train[1000:, 1]
        coordi_train_pt_x = coordi_train[:1000, 0]
        coordi_train_pt_y = coordi_train[:1000, 1]

        coordi_container_pt_x = [[] for _ in range(self.K)]
        coordi_container_pt_y = [[] for _ in range(self.K)]
        for ix in range(1000):
            coordi_container_pt_x[self.attr[ix]].append(coordi_train_pt_x[ix])
            coordi_container_pt_y[self.attr[ix]].append(coordi_train_pt_y[ix])

        colors = ['0.7', 'g', 'b', 'c', 'm', 'y', 'k', '0.1', '0.6', '0.9']
        for ix in range(self.K):
            plt.scatter(coordi_container_pt_x[ix], coordi_container_pt_y[ix], marker='.', c=colors[ix], label='1', s=10)
        plt.scatter(coordi_train_cl_x, coordi_train_cl_y, marker='x', c='r', label='1', s=100)
        plt.show()




if __name__ == '__main__':
    # hyper param
    K = 10
    dist_mode = 'Euclidean'  # Manhattan or Euclidean
    # dist_mode = 'Manhattan'  # Manhattan or Euclidean
    Iteration = 10

    kmeans = KMeans(K, dist_mode, Iteration)
    kmeans.compute()
    kmeans.draw_curve()
    kmeans.draw_distribution()

import numpy as np
import matplotlib.pyplot as plt
from PCA.read_minst import DataLoader

class PCA():
    def __init__(self, dim = 20):
        self.dim = dim

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

    def PCA_reconstruction(self, data):
        '''
            data.shape -> (n, 784)
            output.shape -> (n, 2)
        '''
        # print(data.shape)
        data_mean = np.mean(data, 1)[:, np.newaxis]
        data -= data_mean
        # data /= np.std(data, 1)[:, np.newaxis]
        data_t = data.transpose()
        data_cov = np.cov(data_t)
        evals, evecs = np.linalg.eig(data_cov)
        return np.dot(np.dot(data, evecs[:, :self.dim]), evecs[:, :self.dim].T) + data_mean

    def demo(self):
        plt.imshow(self.trainimg[0].reshape([28, 28]), cmap='Greys_r')
        plt.show()

        pic = self.PCA_reconstruction((self.trainimg[:1000]))[0].reshape([28, 28]).astype(np.float32)
        # print(pic.dtype)
        plt.imshow(pic, cmap='Greys_r')
        plt.show()


if __name__ == '__main__':

    pca = PCA(dim=200)
    pca.demo()
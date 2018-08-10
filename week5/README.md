# week5 PCA and K-Means

### **Result is here**

## PCA Reconstruction

Origin

![](./pic/WechatIMG3.png)

Dim = 20

![](./pic/WechatIMG4.png)

Dim = 100

![](./pic/WechatIMG9.png)

Dim = 200

![](./pic/WechatIMG10.png)



## K-Means Cluster

### Training analyse

LOSS Curve

![](./pic/WechatIMG14.png)

Train and Validation Accuracy

![](./pic/WechatIMG13.png)

The blue one is Val curve, the other is Train curve.

Why Train ACC is lower than Val ACC, my opinion is that training dataset has a great number of images but validation dataset doesn't.

## Scatter plot

**PCA** I first using the PCA algorithm to reduct dimension,  obtain top 2 dim and plot

![](./pic/WechatIMG12.png)

as you can see, the result is poor,  I think that the PCA is not working in dataset with class label, so I change the alogrithm to **TSNE** which is also a alogrithm for dimensionally reduction.

![](./pic/WechatIMG11.png)
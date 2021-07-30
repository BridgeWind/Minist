import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        #大端模式II表示两个unsigned Int
        magic, m = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

        #避免memeory erro
        testImage=images[400:801,:]
        testLabels=labels[400:801]
        images=images[1:401,:]
        labels=labels[1:401]
        m=400

    return images/255*0.9+0.01, labels,m,testImage/255*0.9+0.01,testLabels

images,y,m,testImage,testLabels = load_mnist("D:/A1Study/pytorchWorkshop/Minist/mnistData/")

fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(10):
    # images[y==i]得出images里面所有包含label = i 的数组集合 [0]取第一个
    img = images[y == i][0].reshape(28, 28)
    #img = images[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

# print(m)
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# # plt.tight_layout()
# plt.show()
def reshapeLabel(y,m):
    new_y=np.zeros((m,10))
    for i in range(m):
        new_y[i,y[i]]=1
    return new_y


def sigmoid(array):
    return 1/(1+exp(-array))

def sigmoidGD(array):
    return sigmoid(array)*(1-sigmoid(array))


def theta_initializer(nextLayerSize,lastLayerSize,epsilon):
    # 让theta的值趋于很小0<theta<2*epsilon
    return np.random.rand(nextLayerSize,lastLayerSize+1)*2*epsilon-epsilon

def forwardProp(theta,array):
    r=np.size(array,0)
    #add bias to input array, first column
    _zeros=np.zeros((r,1))
    previous_a = np.c_[_zeros,array]
    z = np.dot(theta,previous_a.T)
    output = sigmoid(z)
    return output,z,previous_a

def predict(theta1,theta2,image):
    output_l2,z2,a1=forwardProp(theta1,image)
    result,z3,a2=forwardProp(theta2,output_l2.T)
    return result.T

def costFun(y,images,theta1,theta2,lambda_,alpha=0.2):   
    #   z2=400*785
    #   a2=theta1*a1.T=25*400 
    output_l2,z2,a1=forwardProp(theta1,images)
    output_l3,z3,a2=forwardProp(theta2,output_l2.T)
    y=reshapeLabel(y,m)
    #   output=400*10
    output=output_l3.T
    J=1/m*((-y*np.log(output)).sum(axis=1)-((1-y)*np.log(1-output)).sum(axis=1)).sum()
    #regularize
    J+=lambda_/(2*m)*((theta1*theta1).sum()+(theta2*theta2).sum())
    #   delta3 400*10
    delta3=output-y
    #   不要偏置项theta2[:,1:].T 25*10 delta3.T 10*400 
    #delta2=25*400
    delta2=np.dot(theta2[:,1:].T,delta3.T)*sigmoidGD(z2)
    #delta3 400*10 a2 400*26
    #10*26
    gradient_l2 = alpha*np.dot(delta3.T,a2)/m
    #delta2=25*400 a1 400*785
    #25*785
    gradient_l1 = alpha*np.dot(delta2,a1)/m
    #Regularize
    gradient_l1+=np.c_[np.zeros((theta1.shape[0],1)),lambda_/m*theta1[:,1:]]
    gradient_l2+=np.c_[np.zeros((theta2.shape[0],1)),lambda_/m*theta2[:,1:]]
    return J,gradient_l1,gradient_l2

# theta1 25*785 
theta1=theta_initializer(30,784,0.12)
# theta2 10*26
theta2=theta_initializer(10,30,0.12)
maxIter=400
for i in range(maxIter):
    J,g1,g2=costFun(y,images,theta1,theta2,0.02)
    theta1-=g1
    theta2-=g2
    # print('学习step:   '+str(i)+'\nCost value J:  '+str(J))

predict=predict(theta1,theta2,testImage)
#computes the index of maxmum number in each row 
result=np.argmax(predict,axis=1)
accuracy=result==testLabels
# 0.7905236907730673
print(sum(accuracy==1)/len(accuracy))










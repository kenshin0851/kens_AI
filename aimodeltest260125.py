import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)


import matplotlib.pyplot as plt
 
image, label = train_dataset[0]
plt.imshow(image[0]) # image의 size가 (1, 28, 28)이므로 (28, 28)으로 변환하여 이미지를 출력한다.
 

import cv2
import numpy
k=image[0]
n=k.detach().cpu().numpy()
#cv2.imshow("frame",n)
n*100
n100=n*100
n100=numpy.array(n100,dtype=int)
#cv2.imshow("frame",n)
plt.imshow(n100)
b=numpy.ones((30,30))
kk=numpy.kron(n100,b)
cv2.imshow("frame",kk)
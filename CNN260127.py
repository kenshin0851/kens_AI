#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:40:47 2026

@author: kensMACbook
"""

import torch
import torch.nn as nn  # 딥러닝 모듈
import torch.nn.init as init
import torch.optim as optim

from torchvision import datasets #FashionMNIST 데이터셋 사용
import torchvision.transforms as transforms # 데이터셋 변형
from torch.utils.data import Dataset, DataLoader # 데이터셋 로딩, 전처리, 순회

batch_size = 100
learning_rate = 0.001
num_epoch = 5

# torchvision의 FashionMNIST 데이터셋 텐서 형태로 불러오기
train_dataset = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = datasets.FashionMNIST(
    root='data', train=False, download=True, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class CNN(nn.Module):
    def __init__(self) : 
        super(CNN,self).__init__() 
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,5), # 이미지 크기: 28x28 -> 24x24
            nn.ReLU(),
            nn.Conv2d(16,32,5), # 이미지 크기: 24x24 -> 20x20
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 이미지 크기: 20x20 -> 10x10
            nn.Conv2d(32,64,5), # 이미지 크기: 10x10 -> 6x6
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 이미지 크기: 6x6 -> 3x3
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size, -1) # 전결합층을 위해서 Flatten하는 과정. 
        out = self.fc_layer(out)
        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.mps.is_available() else "cpu")
model = CNN()
#model.load_state_dict(torch.load('cnn_weights.pth')) #학습한 모델 불러오기
model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

loss_arr = []
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device) # mnist 학습용 data(28x28)
        y_ = label.to(device) # 각각의 data들이 0~9중 어떤숫자인지
        
        optimizer.zero_grad() #optimizer 초기화
        output = model.forward(x) # CNN 학습 시작.
        
        loss = loss_func(output,y_) #학습해서 추정해낸 값과, 실제 라벨된 값 비교
        loss.backward() #오차만큼 다시 Back Propagation 시행
        optimizer.step() #Back Propagation시 ADAM optimizer 매 Step마다 시행
        
        if j % 1000 == 0 : # 1000 미니 배치마다 출력.
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())
            
correct = 0
total = 0
with torch.no_grad(): # 학습을 진행하지 않을 것이므로 torch.no_grad()
    for image,label in test_loader : 
        x = image.to(device)
        y_ = label.to(device)
        output = model.forward(x)
        _,output_index = torch.max(output,1)
        total += label.size(0)
        correct += (output_index == y_).sum().float()
    print("Accuracy of Test Data : {}".format(100*correct/total))
    
# 모델 파라미터 저장
#torch.save(model.state_dict(), 'cnn_weights.pth')  
    
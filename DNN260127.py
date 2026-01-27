#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:31:25 2026

@author: kensMACbook
"""

import torch
import torch.nn as nn  # 딥러닝 모듈
from torch.autograd import Variable # 자동 미분
import torch.nn.functional as F # 활성화 함수

from torchvision import datasets# FashionMNIST 데이터셋 사용
import torchvision.transforms as transforms # 데이터셋 변형
from torch.utils.data import Dataset, DataLoader # 데이터셋 로딩, 전처리, 순회

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.mps.is_available() else "cpu")

# torchvision의 FashionMNIST 데이터셋 텐서 형태로 불러오기
train_dataset = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = datasets.FashionMNIST(
    root='data', train=False, download=True, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

# DNN 모델 생성
class FashionDNN(nn.Module): # 클래스 형태의 모델을 만들 때는 nn.Module 상속 필요
  # 초기화 함수
  def __init__(self):
    # Module 상속
    super(FashionDNN, self).__init__() 
    
    # 모델에서 사용할 모듈 정의
    self.fc1 = nn.Linear(in_features=784, out_features=256) 
    self.drop = nn.Dropout2d(0.25) 
    self.fc2 = nn.Linear(in_features=256,out_features=128)
    self.fc3 = nn.Linear(in_features=128,out_features=10)

  # 순전파 함수
  def forward(self, input_data):
    # numpy의 reshape와 비슷, 크기가 (?, 784)인 2차원 텐서로 변경
    out = input_data.view(-1, 784) 
    
    # 앞에서 정의한 모듈 사용하여 연산 수행 후 ReLU 활성화 함수 사용 반복
    out = F.relu(self.fc1(out)) 
    out = self.drop(out)
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out

# 학습 전 loss function, learning rate, optimizer 정의
learning_rate = 0.001
model = FashionDNN()
#model.load_state_dict(torch.load('model_weights.pth')) #학습한 모델 불러오기
model.to(device) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

num_epochs = 5 # 에포크: 학습 횟수
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []


for epoch in range(num_epochs): # 0 ~ 4 반복
    for images, labels in train_loader: # batch size만큼 데이터 가져옴
    	
        # 데이터가 model과 같은 device에서 처리되도록 설정
        images, labels = images.to(device), labels.to(device) 
    	# 자동 미분
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        # 모델에 넣어서 결과값 return
        outputs = model(train)
        # 손실함수 계산
        loss = criterion(outputs, labels)
        # 역전파 전에 gradient zero로
        optimizer.zero_grad()
        # 역전파 수행
        loss.backward()
        # 역전파에서 수집된 gradient로 조정
        optimizer.step()
        # 반복횟수 +1
        count += 1
    
        # 반복이 50, 100,,,이면 성능 평가
        if not (count % 50):    
            total = 0
            correct = 0        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)            
                test = Variable(images.view(100, 1, 28, 28))            
                outputs = model(test)            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()            
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        # 반복이 500, 1000,,,이면 성능 출력
        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
            
            
# 모델 파라미터 저장
#torch.save(model.state_dict(), 'model_weights.pth')            

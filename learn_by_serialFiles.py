import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch import optim

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
        self.fc = nn.Linear(hidden_dim, output_dim) # RNN 모델의 출력 값을 입력으로 받아들이고, 최종 출력 값을 예측하기 위한 선형 레이어를 정의
        # 
        # 가중치와 편향 텐서의 dtype을 double로 변경
        self.fc.weight.data = self.fc.weight.data.double()
        self.fc.bias.data = self.fc.bias.data.double()
        
    def forward(self, x):
        out, hidden = self.rnn(x)
        out = out[:, -1, :]  # 마지막 시퀀스의 결과만 사용
        out = self.fc(out)  # 최종 결과
        return out

# 모델 생성
input_dim = 15 # feature 14 + energy 1
hidden_dim = 64
output_dim = 1
model = RNNModel(input_dim, hidden_dim, output_dim)
seq_len = 5
m_batch_size = 10

# 손실 함수 정의
criterion = nn.MSELoss()
# 평균 제곱 오차 :
# 예측값과 실제 값의 차이를 제곱해서 데이터의 개수로 나눈 거

# 최적화 기법 선택
optimizer = optim.Adam(model.parameters(), lr=0.001)

#불러올 데이터 파일
folder_path = 'data/jan_serialize_24/'
files = sorted(os.listdir(folder_path), key=lambda x:int(x.split('.')[0]))

epochs = 1000
losses = []

for epoch in range(epochs):
    for dataFile in files:
        x_data = torch.load(folder_path+dataFile)  # (15, 1, 15)
        
        # x_data의 마지막 값을 예측 대상인 y_data로 설정
        y_data = x_data[:, :, -1]  # (15, 1)

        # 모델에 x_data를 전달하여 결과를 추론
        output = model(x_data.double())  # (15, 1)
        
        # loss 계산
        loss = criterion(output, y_data)  # output과 y_data의 차이를 손실로 계산

        # 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 모델 평가 모드
model.eval()

test_loss = 0

for dataFile in files:
    x_data = torch.load(folder_path+dataFile)  # (15, 1, 15)
    y_data = x_data[:, :, -1]  # (15, 1)

    output = model(x_data.double())  # (15, 1)
    loss = criterion(output, y_data)

    test_loss += loss.item()

# 전체 데이터셋의 평균 손실값
test_loss /= len(files)

print(f'Test Loss: {test_loss:.4f}')

import matplotlib.pyplot as plt

plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

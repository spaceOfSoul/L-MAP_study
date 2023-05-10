from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import pickle

forder= os.listdir('1월 데이터')

forder.sort()
# print(forder)

# exit()
weather_by_region = {}
for file in forder:
    csv = pd.read_csv('1월 데이터/' + file, encoding='CP949')
    csv = csv.drop(['지점명'], axis=1)
    groups = csv.groupby(csv.columns[0])
    for i in groups:
        if weather_by_region.get(i[0]) is not None:
            weather_by_region[i[0]].append(list(i))
        else:
            weather_by_region[i[0]] = list(i)

type(weather_by_region[310][1])

for i in weather_by_region:
    region_data=weather_by_region[i]
    
    concated_data = region_data[1]
    size = len(region_data)
    for j in range(2,size):
        concated_data = pd.concat([concated_data, region_data[j][1]])
    
    region_data = concated_data.values
    size=len(region_data)
    # print(region_data[567])
    result = []
    sum = np.zeros(14).astype(float)
    
    for j in range(size):
        row_data = region_data[j][2:].astype(float)
        sum += row_data
        if (j+1) % 60 == 0:
            result.append(sum/60)
            sum = np.zeros(14)

    weather_by_region[i]=result
    
powers = []
forder = os.listdir('2022_01_energy/2022_01')

forder=sorted(forder,key=lambda x:int(x.split('.')[0]))
# print(forder)

for file in forder:
    xlsx = pd.read_excel('2022_01_energy/2022_01/'+file, engine='openpyxl', skiprows=range(4))
    xlsx = xlsx.iloc[:-1, :]  #row remove
    powers.append(xlsx)
# powers

# 모든 데이터프레임을 하나의 리스트로 합침.
all_powers = [power for df in powers for power in df.to_numpy()]

combined_powers = pd.DataFrame(all_powers, columns=['Datetime', 'Power'])
powers_np = combined_powers.to_numpy()
powers_np

#삽입할 위치
insert_positions = [0]+[i for i in range(23, len(powers_np), 24)]

num_inserts = len(insert_positions)

# 타임스탬프 하루씩 증가하게 생성
timestamps = [pd.Timestamp('2022-01-01 00:00:00') + pd.Timedelta(days=i) for i in range(num_inserts)]

# [타임스탬프 , 0.0]
insert_values = np.array([[timestamps[i], 0.0] for i in range(num_inserts)], dtype=object)

powers_np = np.insert(powers_np, insert_positions, insert_values, axis=0)

class CustomDataset(Dataset):
    def __init__(self, weather, energy, region):
        self.weather = weather
        self.energy = energy
        self.region = region
    
    def __len__(self):
        return len(self.energy)
    
    def __getitem__(self, idx):
        energy = self.energy[idx][1]
        weather_data = []
        
        #
        region_weather = self.weather[self.region][idx]
        region_weather = np.nan_to_num(region_weather, nan=0)
        weather_data.append(region_weather)
        
        weather_data = torch.tensor(weather_data)
        
        return weather_data, energy

# 678 : 강릉 성산
dataset = CustomDataset(weather_by_region, powers_np,678)
dataloader = DataLoader(dataset, batch_size=15, shuffle=False, drop_last=True)

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

epochs = 1000
losses = []

for epoch in range(epochs):
    for i, (x_data, y_data) in enumerate(dataloader):
        # x_data: (15, 기상데이터 개수(14))
        # y_data: (15, 1)

        for j in range(11):
            # unsqueeze는 차원을 추가해주는 함수
            x_input = x_data[j:j+4].unsqueeze(0)  # (1, 4, 기상데이터 개수(14))
            x_input = x_input.squeeze(2)
            y_input = y_data[j:j+4].unsqueeze(0)  # (1, 4)
            y_input = y_input.unsqueeze(2) # (1, 4, 1)
            y_true = y_data[j+4].unsqueeze(0)  # (1, 1)
            
            # print(x_input)
            # print(y_input)

            combined_data = torch.cat((x_input, y_input), dim=-1)  # (1, 4, 기상데이터 개수(14) + 1)
            # shape 차원이 3개면 각각 batch_size, seq_len, input_size
            
            output = model(combined_data.double())  # 모델에 combined_data 전달하여 결과 추론
            print(f'output:{output}, y : {y_true}')
            # loss 계산
            loss = criterion(output, y_true) # y^, y

            # 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 파라미터 :
            # 학습 가능한 가중치(weight)와 편향(bias)
            # 선형 회귀 모델의 경우 입력 데이터 x와 그에 대한 출력 y가 있음.
            # y = wx + b로 나타낼 때, w하고 b가 각각 가중치, 바이어스
            
    losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


#모델 평가 모드
model.eval()

test_loss = 0

for i, (x_data, y_data) in enumerate(dataloader):
    for j in range(x_data.shape[0]-4):  # 4개씩 그룹화
        x_input = x_data[j:j+4].unsqueeze(0)
        x_input = x_input.squeeze(2)
        y_input = y_data[j:j+4].unsqueeze(0)
        y_input = y_input.unsqueeze(2)
        y_true = y_data[j+4].unsqueeze(0)

        combined_data = torch.cat((x_input, y_input), dim=-1)

        output = model(combined_data.double())
        loss = criterion(output, y_true)

        test_loss += loss.item()

# 전체 데이터셋의 평균 손실값
test_loss /= (len(dataloader) * (x_data.shape[0]-4)) 

print(f'Test Loss: {test_loss:.4f}')

import matplotlib.pyplot as plt

plt.plot(range(epochs),losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import pickle
import torch
from torch import nn
from torch import optim

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

seq_len = 5
batch_size = 11

class CustomDataset(Dataset):
    def __init__(self, weather, energy, region, seq_len):
        self.weather = weather
        self.energy = energy
        self.region = region
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.energy)-self.seq_len
    
    def __getitem__(self, idx):
        energy = self.energy[idx+self.seq_len][1]
        weather_data = []
        
        for i in range(self.seq_len):
            region_weather = self.weather[self.region][idx + i]
            region_weather = np.nan_to_num(region_weather, nan=0)
            weather_data.append(region_weather)

        # energy를 numpy 배열로 변환하고 차원을 추가
        energy = np.array([energy])
        weather_data = torch.tensor(weather_data)
        
        return weather_data, energy

# 678 : 강릉 성산
dataset = CustomDataset(weather_by_region, powers_np,678, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

for i, (x_data, y_data) in enumerate(dataloader):
    torch.save(x_data,f'data/jan_serialize_{batch_size}/{i}.pt')
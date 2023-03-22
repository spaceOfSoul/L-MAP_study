from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

df = pd.DataFrame()
forder= os.listdir('1월 데이터')

for file in forder:
    csv = pd.read_csv('1월 데이터/'+file, encoding='CP949')
    df = pd.concat([df,csv])

df.set_index(['지점', '지점명', '일시'],inplace=True)
data = df.values.astype(np.float32)

# print(data)

#데이터 객체
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx, :-1]

dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

for i in dataloader:
    print(i.shape)
    

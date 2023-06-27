import numpy as np
import pandas as pd
import os
import glob
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim


#===================================== Weather and Polar voltaic Dataset =================================#
class WPD(Dataset):
	def __init__(self, weather_list, energy_list, region_ID):
		self.wlist = weather_list	# all files for weather info
		self.elist = energy_list	# all files for power gener.
		self.rID   = region_ID
		
	def __len__(self):
		return len(self.wlist)

	def __getitem__(self, idx):
		wfilepath = self.wlist[idx]
		efilepath = self.elist[idx]

		############## weather data loading  #################
		# Loading: weather data for all regions written by chy
		csv = pd.read_csv(wfilepath, encoding='CP949')
		csv = csv.drop(['지점명'], axis=1)
		groups = csv.groupby(csv.columns[0])
		weather_by_region = {}
		for i in groups:
			if weather_by_region.get(i[0]) is not None:
				weather_by_region[i[0]].append(list(i))
			else:
				weather_by_region[i[0]] = list(i)

		# Choose region & Time alignment
		rid = self.rID
		region_data = weather_by_region[rid]
		region_data = region_data[1].values
		weather_data= np.zeros([1440, 15])	# hard coding for 1 day, 14 features & time
		timeflag    = np.ones(1440)
		for i in range(len(region_data)):
			timestamp = region_data[i][1]
			date_, time_ = timestamp.split(' ')
			data = region_data[i][2:].astype(float)
			data = np.nan_to_num(data, nan=0)

			hh = int(time_[:2])
			mm = int(time_[-2:])
			idx = hh*60+mm - 1

			weather_data[idx,0] = idx
			weather_data[idx,1:] = data
			timeflag[idx] = 0

		# interpolation for missing data
		idx = np.where(timeflag==1)[0]
		indices, temp = [], []
		if len(idx) == 1:
			indices.append(idx)
		else:
			diff = np.diff(idx)
			for i in range(len(diff)):
				temp.append(idx[i].tolist())
				if diff[i] == 1:
					temp.append(idx[i+1])
				else:
					indices.append(np.unique(temp).tolist())
					temp = []
			if len(temp) > 0:	# add the last block
				indices.append(temp)
				temp = []

		for n in range(len(indices)):
			idx = indices[n]
			maxV, minV = np.max(idx).astype(int), np.min(idx).astype(int)
			prev = weather_data[minV-1,:]
			post = weather_data[maxV+1,:]
			
			nsteps = len(idx)
			for i in range(nsteps):
				weather_data[i+minV] = (nsteps-i)*prev/(nsteps+1) + (i+1)*post/(nsteps+1)

		#np.save('data.npy', weather_data)
		#np.save('idx.npy', timeflag)
		weather_data = torch.tensor(weather_data)

		############## energy data loading  #################
		# Loading: power generation data written by chy
		xlsx = pd.read_excel(efilepath, engine='openpyxl', skiprows=range(4))
		xlsx = xlsx.iloc[:-1,:]	# row remove
		power = xlsx.to_numpy()
		power = pd.DataFrame(power, columns=['Datetime', 'Power'])
		power_data = power.to_numpy()
		power_data = power_data[:,1].astype(float)
		power_data = torch.tensor(power_data)

		return weather_data, power_data


class PredModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(PredModel, self).__init__()
		self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, dtype=torch.double)
		self.fc = nn.Linear(hidden_dim, output_dim)

		self.fc.weight.data = self.fc.weight.data.double()
		self.fc.bias.data = self.fc.bias.data.double()
        
	def forward(self, x):
		out, hidden = self.rnn(x)
		out = out[:, -1, :]
		out = self.fc(out)
		return out



#=======================================================================================================#

# weather_list = ['../../DB/weather/minute/AWS/1월/aws_gwd_20220101.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220102.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220103.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220104.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220105.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220106.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220107.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220108.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220109.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220110.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220111.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220112.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220113.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220114.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220115.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220116.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220117.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220118.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220119.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220120.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220121.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220122.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220123.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220124.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220125.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220126.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220127.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220128.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220129.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220130.csv.csv', '../../DB/weather/minute/AWS/1월/aws_gwd_20220131.csv.csv']
weather_list = glob.glob('1월 데이터/*')
print(weather_list)
# energy_list = ['../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/1.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/2.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/3.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/4.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/5.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/6.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/7.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/8.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/9.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/10.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/11.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/12.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/13.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/14.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/15.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/16.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/17.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/18.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/19.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/20.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/21.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/22.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/23.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/24.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/25.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/26.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/27.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/28.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/29.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/30.xlsx', '../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/31.xlsx']
energy_list = glob.glob('2022_01_energy/2022_01/*')
#weather_list = ['../../DB/weather/minute/AWS/1월/aws_gwd_20220106.csv.csv']
#energy_list = ['../../DB/NRE/polar_voltaic_GWNU_C9/2022_01/1.xlsx']

dset = WPD(weather_list, energy_list, 678)
dloader = DataLoader(dset, batch_size=1, shuffle=False, drop_last=True)

input_dim = 15 # feature 14 with time
hidden_dim = 64
output_dim = 1

model = PredModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

max_epoch = 1000
seqLeng = 30
nBatch  = 23
prev_data = torch.zeros([seqLeng, input_dim])	# 14 is for featDim
losses = []
for epoch in range(max_epoch):
	loss = 0
	for i, (x, y) in enumerate(dloader):
		x = x.squeeze()
		x = torch.cat((prev_data, x), axis=0)
		prev_data = x[-seqLeng:,:]
		y = y.squeeze()

		nLeng, nFeat = x.shape
		batch_data = []
		for j in range(nBatch):
			stridx = j*60
			endidx = j*60 + seqLeng
			batch_data.append(x[stridx:endidx,:].view(1,seqLeng, nFeat))
		batch_data = torch.cat(batch_data, dim=0)

		output = model(batch_data)
		loss += criterion(output.squeeze(), y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	losses.append(loss.item())
	print(f'Epoch [{epoch+1}/{max_epoch}], Loss: {loss.item():.4f}')




	


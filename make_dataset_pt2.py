import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

#setting


# Define the generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, masked_x, mask):
        inp = torch.cat([masked_x, mask], dim=1)
        out = self.fc1(inp)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fake_data, hint):
        #global global_cek
        inp = torch.cat([fake_data, hint], dim=1)
        out = self.fc1(inp)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Set hyperparameters
#input_size = 70
hidden_size = 10
hidden_size = 150
#output_size = 35
num_epochs = 10000
batch_size = 64
#lr = 0.001
lr = 0.000001
train_ratio = 0.7
hint_percentage = 0.9

row_limit = 5

df = pd.read_excel('wiii_dset_qc.xlsx')
col_used = df.loc[:, ['temperature', 'dew', 'vis', 'wind_direction', 'wind_speed', 'pressure']]
col_used_arr = np.array(col_used)
print(col_used_arr)
print(np.shape(col_used_arr))

dset_arr = []
for i in range(len(col_used_arr)-row_limit):
    single_arr = col_used_arr[i:i+row_limit]
    if np.isnan(single_arr).any():
        continue
    if len(dset_arr) == 0:
        dset_arr = single_arr
    elif len(np.shape(dset_arr)) == 3:
        single_arr_3d = np.reshape(single_arr, (1, np.shape(single_arr)[0], np.shape(single_arr)[1]))
        dset_arr = np.vstack((dset_arr, single_arr_3d))
        print('cek shape stack', np.shape(dset_arr))
        #exit()
    else:
        dset_arr = np.stack((dset_arr, single_arr))
        print('cek shape stack', np.shape(dset_arr))
        

np.save('dset_imputation.npy', dset_arr)
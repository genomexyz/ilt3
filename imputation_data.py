import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

#setting
hidden_size = 10
hidden_size = 150
#output_size = 35
num_epochs = 10000
batch_size = 64
#lr = 0.001
lr = 0.000001
train_ratio = 0.7
hint_percentage = 0.9

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

dset_raw = np.load('dset_imputation.npy')

mask_mat = np.random.rand(len(dset_raw), len(dset_raw[0]), len(dset_raw[0,0]))
mask_mat[mask_mat > 0.2] = 0 #0 for non missing
mask_mat[mask_mat > 0] = 1 #1 for missing

randomizer = np.arange(len(dset_raw))
np.random.shuffle(randomizer)
idx_train = randomizer[:int(len(randomizer)*train_ratio)]
idx_test = randomizer[int(len(randomizer)*train_ratio):]

dset_train = dset_raw[idx_train]
mask_train = mask_mat[idx_train]

dset_test = dset_raw[idx_test]
mask_test = mask_mat[idx_test]

print(mask_test)
exit()

dset_train_torch = torch.from_numpy(dset_train)
mask_train_torch = torch.from_numpy(mask_train)

dset_test_torch = torch.from_numpy(dset_test)
mask_test_torch = torch.from_numpy(mask_test)

max_torch = dset_train_torch.max(dim=0)[0].max(dim=0)[0]
min_torch = dset_train_torch.min(dim=0)[0].min(dim=0)[0]

#print('cek max min')
#print(max_torch)
#print(min_torch)
#exit()

expanded_max = max_torch.view(1, 1, max_torch.shape[0])
expanded_min = min_torch.view(1, 1, min_torch.shape[0])

dset_train_torch_norm = (dset_train_torch - expanded_min) / (expanded_max - expanded_min)
dset_test_torch_norm = (dset_test_torch - expanded_min) / (expanded_max - expanded_min)

mean_torch = dset_train_torch_norm.mean(dim=(0,1))

dset_norm_norm_real = torch.zeros((batch_size, dset_train_torch_norm.size()[1], dset_train_torch_norm.size()[2]))
dset_norm_norm_real[:] = mean_torch
dset_norm_norm_real = torch.reshape(dset_norm_norm_real, (batch_size, -1))

dataset_train = TensorDataset(dset_train_torch_norm, mask_train_torch)
dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

print(dset_train_torch_norm, dset_train_torch_norm.size())
print(dset_norm_norm_real)
print(mean_torch)


# Initialize the models
input_size = dset_norm_norm_real.size()[1] * 2
output_size = dset_norm_norm_real.size()[1]

generator = Generator(input_size, hidden_size, output_size).double()
discriminator = Discriminator(input_size, hidden_size, output_size).double()

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Train the GAN
loss_gen_real = nn.MSELoss()
for epoch in range(100):
    for i, (x, m) in enumerate(dataloader):
        x_flat = torch.reshape(x, (x.size()[0], -1))
        m_flat = torch.reshape(m, (m.size()[0], -1))
        dset_norm_norm = dset_norm_norm_real[:len(x)]
        masked_input = x_flat * (1-m_flat) + dset_norm_norm * m_flat
        #print(masked_input, masked_input.size())
        #print(m)
        optimizer_d.zero_grad()
        fake_data = generator(masked_input, m_flat)
        fake_data = fake_data.detach()
        hint_leak_chance = torch.rand(fake_data.size())
        hint_leak = torch.clone(m_flat)
        hint_leak[hint_leak_chance > hint_percentage] = 0.5
        eval_d = discriminator(fake_data, hint_leak)
        #loss_d = -torch.mean(m_flat * torch.log(eval_d) + (1 - m_flat) * torch.log(1 - eval_d))
        loss_d = -torch.mean(m_flat * torch.clamp(torch.log(eval_d), min=0) + (1 - m_flat) * torch.clamp(torch.log(1 - eval_d), min=-1.0))
        loss_d.backward()
        optimizer_d.step()

        # compute generator loss
        optimizer_g.zero_grad()
        fake_data = generator(masked_input, m_flat)
        eval_d = eval_d.detach()
        #lg = -(m_flat * torch.log(eval_d))
        lg = -(m_flat * torch.clamp(torch.log(eval_d), min=0))
        lm = loss_gen_real(m_flat * fake_data, m_flat * x_flat)
        loss_g = torch.mean(lg + lm)
        #print(loss_g)
        loss_g.backward()
        optimizer_g.step()

        if (i+1) % 100 == 0:
            #print(output, y)
            print("Epoch: {}, Iteration: {}, Loss generator: {:.8f}, Loss discriminator: {:.8f}".format(epoch+1, i+1, 
                                                                                                        loss_g.item(),
                                                                                                        loss_d.item()))
    if epoch % 10 == 0:
        checkpoint = {'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                'max': max_torch,
                'min': min_torch,
                'mean_norm': dset_norm_norm_real,
                'norm_train_feature' : dset_train_torch_norm,
                'norm_test_feature': dset_test_torch_norm,
                'train_mask': mask_train_torch,
                'test_mask': mask_test_torch}
        torch.save(checkpoint, "model_epoch{}_gainhomemade.pt".format(epoch+1))

checkpoint = {'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
        'max': max_torch,
        'min': min_torch,
        'mean_norm': dset_norm_norm_real,
        'norm_train_feature' : dset_train_torch_norm,
        'norm_test_feature': dset_test_torch_norm,
        'train_mask': mask_train_torch,
        'test_mask': mask_test_torch}
torch.save(checkpoint, "model_final_gainhomemade-qc.pt")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=4)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=4)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv4 = nn.Conv2d(20, 40, kernel_size=(10,5))
        self.fc1 = nn.Linear(40, 40)
        self.fc2 = nn.Linear(40, 20)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.defc2 = nn.Linear(20, 40)
        self.defc1 = nn.Linear(40, 40)
        self.deconv4 = nn.ConvTranspose2d(40, 20, kernel_size=(10,5))
        self.deconv3 = nn.ConvTranspose2d(20, 20, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(20, 20, kernel_size=7, stride=4)
        self.deconv1 = nn.ConvTranspose2d(20, 3, kernel_size=8, stride=4)
        
    def forward(self, x):
        x = F.relu(self.defc2(x))
        x = F.relu(self.defc1(x))
        x = x.view(-1, 40, 1, 1)
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv1(x))
        return x
        
if __name__ == '__main__':
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    arrs = np.load('./training_data/0/0.npz')
    scrs = arrs['screens'] / 256
    ctrls1 = arrs['controls1']
    ctrls2 = arrs['controls2']
    rms = arrs['rooms']
    l1_crit = nn.L1Loss(reduction='mean')
    for i in range(10000):
        inds = np.random.choice(np.arange(1, 200), 4)
        x = torch.Tensor(scrs[inds])
        feat, ima = model(x)
        #ima = ima.detach().numpy()
        loss = l1_crit(x, ima)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(scrs.shape)
        if i % 10000 == 9999:
            for j in range(4):
                tmp1 = np.transpose(ima[j].detach().numpy(), (2, 1, 0))
                tmp2 = np.transpose(scrs[inds[j]], (2, 1, 0))
                plt.imshow(tmp1)
                plt.show()
                plt.imshow(tmp2)
                plt.show()
            
    

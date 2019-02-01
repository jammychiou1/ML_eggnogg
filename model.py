import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(76, 20, kernel_size=5, stride=4)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=4)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv4 = nn.Conv2d(20, 40, kernel_size=(10,5))
        self.fc1 = nn.Linear(40, 40)
        self.q = nn.Linear(40, 36)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(-1, 40)
        x = F.relu(self.fc1(x))
        return self.q(x)
        
if __name__ == '__main__':
    model = Model()
    a = torch.randn(1, 3, 240, 160)
    print(a)
    print(model(a))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(229, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 40)
        self.fc4 = nn.Linear(40, 36)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = 20 * torch.torch.tanh(self.fc4(x))
        return x
            
    

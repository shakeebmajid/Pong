
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyModel(nn.Module):
    def __init__(self,learning_rate=1e-4):
        super(PolicyModel, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.conv3 = nn.Conv2d(3, 9, 3)
        # an affine operation: y = Wx + b
        self.fcHidden = nn.Linear(9 * 18 * 18, 100)
        self.fcOut = nn.Linear(100, 4) 
        self.softmax = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x.float())) 
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x.float())) 
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x.float())) 
        x = F.max_pool2d(x, (2, 2))
        # If the size is a square you can only specify a single number
        # print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = self.fcHidden(x)
        x = F.relu(x)
        x = self.fcOut(x)
        x = self.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_action(self, state):
        probs = self.forward(state)
        # print(probs)
        highest_prob_action = np.random.choice([0, 1, 2, 3], p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

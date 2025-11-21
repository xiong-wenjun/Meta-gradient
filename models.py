"""models for a2c agents"""
import torch
from torch import nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, l_obs, n_act, hidden_size=32):
        super(ActorCritic, self).__init__()
        self.l_obs = l_obs
        self.n_act = n_act
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(self.l_obs, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.actor = nn.Linear(self.hidden_size, self.n_act)
        self.critic = nn.Linear(self.hidden_size, 1)

        self.fc.apply(weight_init)
        self.actor.apply(weight_init)
        self.critic.apply(weight_init)

    def forward(self, inputs):
        x = self.fc(inputs)
        pi = self.actor(x)
        vi = self.critic(x)

        return pi, vi
    

class ConvActorCritic(nn.Module):
    def __init__(self, l1, l2, l_obs, n_action):
        super(ConvActorCritic, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.l_obs = l_obs
        self.n_action = n_action
        self.conv = nn.Sequential(
            nn.Conv2d(l_obs, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(1024, 512),
                                nn.ReLU())
        self.act_net = nn.Linear(512, self.n_action)
        self.cri_net = nn.Linear(512, 1)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.act_net(x), self.cri_net(x)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu')
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.zeros_(param.data)
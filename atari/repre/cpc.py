
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, one_hot, relu

from dqn.model import mnih_cnn
from dqn.prepare_obs import prepare_obs
from common.optim import ParamOptim
from dqn.buffer import Buffer


class CPCModel(nn.Module):
    def __init__(self, num_action, size_emb, size_stack, device="cuda"):
        super(CPCModel, self).__init__()
        self.size_emb = size_emb
        self.size_stack = size_stack
        self.num_action = num_action
        self.device = device

        self.conv = mnih_cnn(size_stack, size_emb)
        self.rnn = nn.GRUCell(num_action + size_emb, 512)
        self.fc = nn.Linear(512, size_emb)

    def forward(self, obs, action, done, hx=None, only_hx=False):
        obs = prepare_obs(obs, done, self.size_stack)
        steps, batch, *img_shape = obs.shape
        obs = obs.view(steps * batch, *img_shape)
        z = self.conv(obs).view(steps, batch, self.size_emb)

        pf = self.size_stack - 1
        mask = (1 - done[pf:]).float()
        a = one_hot(action[:, :, 0], self.num_action).float()
        x = torch.cat([relu(z[:-1]), a], 2)

        steps -= 1
        y = torch.empty(steps, batch, 512, device=self.device)
        for i in range(steps):
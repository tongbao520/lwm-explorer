import random
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from common.optim import ParamOptim
from dqn.prepare_obs import prepare_obs


class PredictorModel(nn.Module):
    def __init__(self, num_action, fstack, emb_size, rnn_size):
        super(PredictorModel, self).__init__()
        self.fstack = fstack
        self.num_action = num_action
        self.rnn_size = rnn_size
        self.emb_fc = nn.Linear(emb_size + num_action, 128)
        self.rnn = nn.GRUCell(128, rnn_size)
        self.fc = nn.Sequential(
            nn.Linear(rnn_size, rnn_size), nn.ReLU(), nn.Linear(rnn_size, emb_size),
        )

    def forward(self, z, action, done, hx=None):
        unroll, batch, emb_size = z.shape
        a = one_hot(action[:, :, 0], self.num_action).float()
        z = torch.cat([z, a], dim=2)
        z = self.emb_fc(z.view(unroll * batch, (emb_size + self.num_action)))
        z = z.view(unroll, batch, 128)

        mask = 1 - done.float()
        x = torch.empty(unroll, batch, self.rnn_size, device=z.device)
        for i in range(unroll):
            if hx is not None:
                hx *= mask[i]
           
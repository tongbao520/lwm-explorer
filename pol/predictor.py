import random
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from common.optim import ParamOptim


class PredictorModel(nn.Module):
    def __init__(self, rnn_size):
        super(PredictorModel, self).__init__()
        self.rnn_size = rnn_size
        self.encoder = nn.Sequential(nn.Linear(4 + 4, 32), nn.ReLU())
        # self.rnn = nn.GRUCell(32, self.rnn_size)
        self.rnn = nn.GRU(32, self.rnn_size)
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, 4),
            nn.Sigmoid(),
        )

    def forward(self, z, action, done, hx=None):
        unroll, batch, emb_size = z.shape
        a = one_hot(action[:, :, 0], 4).float()
        z = torch.cat([z, a], dim=2)
        z = self.encoder(z.view(unroll * batch, 4 + 4))
        z = z.view(unroll, batch, 32)

        # mask = 1 - done.float()
        # x = torch.empty(unroll, batch, self.rnn_size, device=z.device)
        # for i in range(unroll):
        #     if hx is not None:
        #         hx *= mask[i]
        #     x[i] = hx = self.rnn(z[i], hx)
        # hx = hx.clone().detach()

        x, hx = self.rnn(z, hx)

        x = self.fc(x.view(unroll * batch, self.rnn_size))
        z_pred = x.view(unroll, batch, 4)
        return z_pred, hx


class Predictor:
    def __init__(self, buffer, cfg, device="cuda"):
        self.d
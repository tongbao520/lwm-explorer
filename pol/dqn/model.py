import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class DQN(nn.Module):
    def __init__(self, rnn_size, device="cuda"):
        super(DQN, self).__init__()
        self.device = device
        self.rnn_size = rnn_size

        self.encoder = nn.Sequential(nn.Linear(4 + 4 + 1, 32), nn.ReLU())
        # self.rnn = nn.GRUCell(32, self.rnn_size)
        self.rnn = nn.GRU(32, self.rnn_size)
        self.adv = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, 4, bias=False),
        )
        self.val = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, 1),
        )

    def forward(self, obs, action, reward, done, hx=None, only_hx=False):
        mask = (1 - done).float()
        a = one_hot(action[:, :, 0], 4).float() * mask
        r = reward * mask
        x = torch.cat([obs.float(), a, r], 2)

        steps, batch, *rest = x.shape
        x = x.view(steps * batch, *rest)
        x = self.encoder(x).view(steps, batch, 32)

        # y = torch.emp
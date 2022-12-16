
import random
import torch
from common.optim import ParamOptim
from repre.whitening import Whitening2d
from dqn.model import mnih_cnn
from dqn.prepare_obs import prepare_obs


class WMSE:
    batch_size: int = 256
    lr: float = 5e-4

    def __init__(self, buffer, cfg, device="cuda"):
        self.device = device
        self.buffer = buffer
        self.emb_size = cfg["w_mse"]["emb_size"]
        self.temporal_shift = cfg["w_mse"]["temporal_shift"]
        self.spatial_shift = cfg["w_mse"]["spatial_shift"]
        self.frame_stack = cfg["w_mse"]["frame_stack"]

        self.encoder = mnih_cnn(self.frame_stack, self.emb_size)
        self.encoder = self.encoder.to(self.device).train()
        self.optim = ParamOptim(lr=self.lr, params=self.encoder.parameters())
        self.w = Whitening2d(self.emb_size, track_running_stats=False)

    def load(self):
        cp = torch.load("models/w_mse.pt", map_location=self.device)
import torch


class DictWithSlicing(dict):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return {k: v[key] for k, v in self.items()}
        return super().__getitem__(key)


class Buffer:
    def __init__(self, maxlen, num_env, obs_shape, device):
        self.maxlen, self.num_env, self.device = maxlen, num_env, device
        self.cursor = self._size = 0

        def tensor(shape=(1,), dtype=torch.float):
            return torch.empty(
                self.maxlen, self.num_env, *shape, dtype=dtype, device=self.device
            )

        self._buffer = {
            "obs": tensor(obs_shape, torch.uint8),
            "action": tensor(dtype=torch.long),
            "reward": tensor(),
            "done": tensor(dtype=torch.uint8),
        }

    def query(self, idx, id
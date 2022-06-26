import torch


class DictWithSlicing(dict):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return {k: v[key] for k, v in self.items()}
        return super().__getitem__(key)


class Buffer:
    def __init__(self, maxlen, num_env, obs_shape, device):
        self.maxlen, self.num_env, self.device = maxlen, num_env, device
        self.reset()

        def tensor(shape=(1,), dtype=torch.float):
            return torch.empty(
                self.maxlen, self.num_env, *shape, dtype=dtype, device=self.device
            )

        self._buffer = {
            "obs": tensor(
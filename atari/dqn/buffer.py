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
            "obs": tensor(obs_shape, torch.uint8),
            "action": tensor(dtype=torch.long),
            "reward": tensor(),
            "done": tensor(dtype=torch.uint8),
        }

    def query(self, idx, idx_env, steps, device="cuda"):
        qsize = len(idx)
        s = torch.arange(steps - 1, -1, -1)
        q0 = (idx[None, ...].repeat(steps, 1) - s[..., None]).flatten()
        q1 = idx_env[None, ...].repeat(steps, 1).flatten()
        return DictWithSlicing(
            {
                k: v[q0, q1].view(steps, qsize, *v.shape[2:]).to(device)
                for k, v in self._buffer.items()
            }
        )

    def append(self, step):
        for k in self._buffer:
            if k not in step:
                self._buffer[k][self.cursor] = 0
            else:
                assert step[k].dtype == self._buffer[k].dtype
                assert step[k].shape == self._buffer[k].shape[1:]
                self._buffer[k][self.cursor] = step[k].to(self.device)
        self.cursor = (self.cursor + 1) % self.maxlen

import torch


class DictWithSlicing(dict):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return {k: v[key] for k, v in self.items()}
        return super().__getitem__(ke
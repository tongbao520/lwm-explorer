import torch


class DictWithSlicing(dict):
    def __getitem__(self, key):
        if isinstance(key
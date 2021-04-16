import torch


class DistributedTrainer:
    def __init__(self, G, base_model):
        self.G = G
        self.base_model = base_model
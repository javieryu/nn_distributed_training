import torch
from torch.optim import Optimizer


class DSGD:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf

    def step(self):
        return 0

    def train(self, profiler=None):
        eval_every = self.pr.conf["evaluate_frequency"]
        oits = self.conf["outer_iterations"]

        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics()

            self.step()

            if profiler is not None:
                profiler.step()
        return 0
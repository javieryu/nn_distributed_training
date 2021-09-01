import torch
from utils import graph_generation
import copy


class DSGT:
    def __init__(self, ddl_problem, device, conf):
        self.pr = ddl_problem
        self.conf = conf
        self.device = device

        # Get list of all model parameter pointers
        self.plists = {
            i: list(self.pr.models[i].parameters()) for i in range(self.pr.N)
        }

        # Useful numbers
        self.num_params = len(self.plists[0])
        self.alpha = conf["alpha"]

        base_zeros = [
            torch.zeros_like(p, requires_grad=False, device=self.device)
            for p in self.plists[0]
        ]
        self.glists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}
        self.ylists = {i: copy.deepcopy(base_zeros) for i in range(self.pr.N)}

    def train(self, profiler=None):
        eval_every = self.pr.conf["metrics_config"]["evaluate_frequency"]
        oits = self.conf["outer_iterations"]

        # Initialize Ylists and Glists
        for i in range(self.pr.N):
            bloss = self.pr.local_batch_loss(i)
            bloss.backward()

            with torch.no_grad():
                for p in range(self.num_params):
                    self.ylists[i][p] = self.plists[i][p].grad.detach().clone()
                    self.glists[i][p] = self.plists[i][p].grad.detach().clone()
                    self.plists[i][p].grad.zero_()

        # Optimization loop
        for k in range(oits):
            if k % eval_every == 0 or k == oits - 1:
                self.pr.evaluate_metrics()

            # Compute graph weights
            W = graph_generation.get_metropolis(ddl_problem.graph)
            W = W.to(self.device)

            # Iterate over the agents for communication step
            for i in range(self.pr.N):
                neighs = list(self.pr.graph.neighbors(i))
                with torch.no_grad():
                    # Update each parameter individually across all neighbors
                    for p in range(self.num_params):
                        # Ego update
                        self.plists[i][p].multiply_(W[i, i])
                        self.plists[i][p].add_(-self.alpha * self.ylists[i][p])
                        # Neighbor updates
                        for j in neighs:
                            self.plists[i][p].add_(W[i, j] * self.plists[j][p])

            # Compute the batch loss and update using the gradients
            for i in range(self.pr.N):
                # Batch loss
                bloss = self.pr.local_batch_loss(i)
                bloss.backward()

                neighs = list(self.pr.graph.neighbors(i))
                # Locally update model with gradient
                with torch.no_grad():
                    for p in range(self.num_params):
                        self.ylists[i][p].multiply_(W[i, i])
                        for j in neighs:
                            self.ylists[i][p].add_(W[i, j] * self.ylists[j][p])

                        self.ylists[i][p].add_(self.plists[i][p].grad)
                        self.ylists[i][p].add_(-1.0 * self.glists[i][p])

                        self.glists[i][p] = (
                            self.plists[i][p].grad.detach().clone()
                        )
                        self.plists[i][p].grad.zero_()

            if profiler is not None:
                profiler.step()
        return

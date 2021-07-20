import torch


class CADMM:
    def __init__(self, ddl_problem, conf):
        self.pr = ddl_problem
        self.conf = conf

        self.duals = {i: torch.zeros(self.pr.n) for i in range(self.pr.N)}

        self.rho = self.conf["rho_init"]
        self.rho_scaling = self.conf["rho_scaling"]
        self.primal_lr = self.conf["primal_lr"]
        self.pits = self.conf["primal_iterations"]

    def primal_update(self, i, neighs, ths):
        if self.conf["inner_optimizer"] == "adam":
            opt = torch.optim.Adam(
                self.pr.models[i].parameters(), self.primal_lr
            )
        elif self.conf["inner_optimizer"] == "sgd":
            opt = torch.optim.SGD(
                self.pr.models[i].parameters(), self.primal_lr
            )
        elif self.conf["inner_optimizer"] == "adamw":
            opt = torch.optim.AdamW(
                self.pr.models[i].parameters(), self.primal_lr
            )
        else:
            raise NameError("CADMM Inner Optimizer is unknown.")

        for _ in range(self.pits):
            opt.zero_grad()

            # Model pass on the batch
            pred_loss = self.pr.local_batch_loss(i)

            # Get the primal variable WITH the autodiff graph attached.
            th = torch.nn.utils.parameters_to_vector(
                self.pr.models[i].parameters()
            ).reshape(-1, 1)

            reg = sum(
                torch.square(
                    torch.norm(th - ths[j].reshape(1, -1)) for j in neighs
                )
            )

            loss = pred_loss + torch.dot(th, self.duals[i]) + self.rho * reg
            loss.backward()
            opt.step()

        return

    def train(self):
        eval_every = self.pr["evaluate_frequency"]
        oits = self.conf["outer_iterations"]
        for k in range(oits):
            # Get the current primal variables
            ths = {
                i: torch.nn.utils.parameters_to_vector(
                    self.pr.models[i].parameters()
                )
                .clone()
                .detach()
                for i in range(self.pr.N)
            }

            # Update the penalty parameter
            self.rho *= self.rho_scaling

            # Per node updates
            for i in range(self.pr.N):
                if k % eval_every == 0 or k == oits - 1:
                    self.pr.evaluate_metrics()

                neighs = list(self.pr.graph.neighbors(i))

                self.duals[i] += self.rho * sum(
                    ths[i] - ths[j] for j in neighs
                )
                self.primal_update(i, neighs, ths)

        return
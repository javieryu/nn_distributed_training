import os
import copy

import torch


class DistMNISTProblem:
    """An object that manages the various datastructures for a distributed
    optimization problem on the MNIST classification problem. In addition,
    it computes, stores, and writes out relevant metrics.

    Author: Javier Yu, javieryu@stanford.edu, July 20, 2021
    """

    def __init__(
        self,
        graph,
        base_model,
        base_loss,
        train_sets,
        val_set,
        conf,
    ):
        self.graph = graph
        self.base_loss = base_loss
        self.train_sets = train_sets
        self.val_set = val_set
        self.conf = conf

        # Extract some useful info
        self.N = graph.number_of_nodes()
        self.n = torch.nn.utils.parameters_to_vector(
            base_model.parameters()
        ).shape[0]

        # Copy the base_model for each node
        self.models = {i: copy.deepcopy(base_model) for i in range(self.N)}

        # Create train loaders and iterators with specified batch size
        self.train_loaders = {}
        self.train_iters = {}
        for i in range(self.N):
            self.train_loaders[i] = torch.utils.data.DataLoader(
                self.train_sets[i],
                batch_size=self.conf["batch_size"],
                shuffle=True,
            )

            self.train_iters[i] = iter(self.train_loaders[i])

        self.val_loader = torch.utils.data.DataLoader(
            self.val_set, batch_size=self.conf["val_batch_size"]
        )

        # Initialize lists for metrics with names
        self.metrics = {met_name: [] for met_name in self.conf["metrics"]}
        self.epoch_tracker = torch.zeros(self.N)
        self.forward_cnt = 0

    def local_batch_loss(self, i):
        """Forward pass on a batch data for model at node i,
        and if it's node_id = 0 then increment a metric that
        counts the number of forward passes. Also increment an
        epoch tracker for each node when the iterator resets.

        Finally compute loss based on base_loss function and return.

        Note: if for whatever reason there isn't a node zero then
        this method's metric increment step will fail.

        Args:
            i (int): Node id

        Returns:
            (torch.Tensor): Loss of node i's model on a batch of
            local data.
        """
        try:
            x, y = next(self.train_iters[i])
        except StopIteration:
            self.epoch_tracker[i] += 1
            self.train_iters[i] = iter(self.train_loaders[i])
            x, y = next(self.train_iters[i])

        if i == 0:
            # Count the number of forward passes that have been performed
            # because this is symmetric across nodes we only have to do
            # this for node 0, and it will be consistent with all nodes.
            self.forward_cnt += self.conf["batch_size"]

        yh = self.models[i].forward(x)

        return self.base_loss(yh, y)

    def save_metrics(self, output_dir):
        """Save current metrics lists to a PT file."""
        file_name = os.path.join(output_dir, "metrics.pt")
        torch.save(self.metrics, file_name)
        return

    def validate(self, i):
        """Compute the loss and accuracy of a
        single node's model on the validation set.

        Args:
            i (int): Node id
        """
        with torch.no_grad():
            loss = 0.0
            correct = 0
            for x, y in self.val_loader:
                yh = self.models[i].forward(x)
                loss += self.base_loss(yh, y).item()
                pred = yh.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
            avg_loss = loss / len(self.val_loader.dataset)
            acc = correct / len(self.val_loader.dataset)
            return avg_loss, acc

    def evaluate_metrics(self):
        """Evaluate models, and then append values to the metric lists."""

        # Compute validation loss and accuracy (if you do one you might
        # as well do the other)
        if (
            "validation_loss" in self.metrics
            or "top1_accuracy" in self.metrics
        ):
            avg_losses = torch.zeros(self.N)
            accs = torch.zeros(self.N)
            for i in range(self.N):
                avg_losses[i], accs[i] = self.validate(i)

        evalprint = "| "
        for met_name in self.conf["metrics"]:
            if met_name == "consensus_error":
                # The average distance from a single node to all
                # of the other nodes in the problem
                with torch.no_grad():
                    all_params = [
                        torch.nn.utils.parameters_to_vector(
                            self.models[i].parameters()
                        )
                        for i in range(self.N)
                    ]
                    # Stack all of the parameters into rows
                    th_stack = torch.stack(all_params)
                    # Compute row-wise distances
                    distances = torch.cdist(th_stack, th_stack)
                    davg = distances.sum(axis=1) / self.N
                # append metrics and generate print string
                self.metrics[met_name].append(distances)
                evalprint += "Consensus: {:.4f} - {:.4f} | ".format(
                    torch.amin(davg).item(), torch.amax(davg).item()
                )
            elif met_name == "validation_loss":
                # Average node loss on the validation dataset
                self.metrics[met_name].append(avg_losses)
                evalprint += "Val Loss: {:.4f} - {:.4f} | ".format(
                    torch.amin(avg_losses).item(),
                    torch.amax(avg_losses).item(),
                )
            elif met_name == "top1_accuracy":
                # Top1 accuracy of nodes on the validation dataset
                self.metrics[met_name].append(accs)
                evalprint += "Top1: {:.2f} - {:.2f} |".format(
                    torch.amin(accs).item(), torch.amax(accs).item()
                )
            elif met_name == "forward_pass_count":
                # Number of forward passes performed by each node
                self.metrics[met_name].append(self.forward_cnt)
                evalprint += "Num Forward: {} | ".format(self.forward_cnt)
            elif met_name == "current_epoch":
                # Current epoch of each node (only different if the datasets at
                # each node are not the same size)
                self.metrics[met_name].append(copy(self.epoch_tracker))
                evalprint += "Ep Range: {} - {} | ".format(
                    int(torch.amin(self.epoch_tracker).item()),
                    int(torch.amax(self.epoch_tracker).item()),
                )
            else:
                raise NameError("Unknown metric.")

        print(evalprint)
        return

import torch
import copy


class DistMNISTProblem:
    def __init__(
        self,
        graph,
        base_model,
        base_loss,
        train_set,
        eval_set,
        conf,
    ):
        self.graph = graph
        self.base_loss = base_loss
        self.train_set = train_set
        self.eval_set = eval_set
        self.conf = conf

        self.N = graph.number_of_nodes()
        self.n = torch.nn.utils.parameters_to_vector(base_model).shape[0]
        self.models = {i: copy.deepcopy(base_model) for i in range(self.N)}

    def forward_batch(self, i, batch_size=None):
        """

        Args:
            i ([type]): [description]
        """
        return

    def create_metrics(self):
        """Create a dictionary of evaluation metrics with empty lists
        based on the configuration specification.
        Metric ideas:
        - Global consensus (all to all)
        - Local consensus (neighbor to neighbor)
        - Validation set loss"""

        return 0

    def metrics_to_vec(self):
        """Convert metric lists to Tensors."""
        return

    def save_metrics(self):
        """Save currrent metrics lists to a PT file."""
        return

    def evaluate_metrics(self):
        """Evaluate models, and then append values to the metric lists."""
        return
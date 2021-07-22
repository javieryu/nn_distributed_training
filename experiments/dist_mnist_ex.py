import os
import sys
from datetime import datetime
from shutil import copyfile

import yaml
from torchvision import datasets, transforms
import torch
import networkx as nx

sys.path.insert(0, "../models/")
sys.path.insert(0, "../problems/")
sys.path.insert(0, "../optimizers/")
from mnist_conv_nn import MNISTConvNet
from dist_mnist_problem import DistMNISTProblem
from cadmm import CADMM


def experiment(yaml_pth):
    # load the config yaml
    with open(yaml_pth) as f:
        conf_dict = yaml.safe_load(f)

    # Seperate configuration groups
    exp_conf = conf_dict["experiment"]

    # Create the output directory
    output_metadir = exp_conf["output_metadir"]
    if not os.path.exists(output_metadir):
        os.mkdir(output_metadir)

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = os.path.join(
        output_metadir, time_now + "_" + exp_conf["name"]
    )

    if exp_conf["writeout"]:
        os.mkdir(output_dir)
        # Save a copy of the conf to the output directory
        copyfile(yaml_pth, os.path.join(output_dir, time_now + ".yaml"))
    exp_conf["output_dir"] = output_dir  # probably bad practice

    # Create communication graph
    graph_conf = exp_conf["graph"]
    N = graph_conf["num_nodes"]
    if graph_conf["type"] == "wheel":
        graph = nx.wheel_graph(N)
    elif graph_conf["type"] == "random":
        # Attempt to make a random graph until it is connected
        graph = nx.erdos_renyi_graph(N, graph_conf["p"])
        for _ in range(graph_conf["gen_attempts"]):
            if nx.is_connected(graph):
                break
            else:
                graph = nx.erdos_renyi_graph(N, graph_conf["p"])

        if not nx.is_connected(graph):
            raise NameError(
                "A connected random graph could not be generated,"
                " increase p or gen_attempts."
            )
    else:
        raise NameError("Unknown communication graph type.")

    if exp_conf["writeout"]:
        # Save the graph for future visualization
        nx.write_gpickle(graph, os.path.join(output_dir, "graph.gpickle"))

    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_dir = exp_conf["data_dir"]
    joint_train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    val_set = datasets.MNIST(data_dir, train=False, transform=transform)

    if exp_conf["data_split_type"] == "random":
        num_samples_per = len(joint_train_set.targets) / N
        joint_splits = [int(num_samples_per) for _ in range(N)]
        train_subsets = torch.utils.data.random_split(
            joint_train_set, joint_splits
        )
    elif exp_conf["data_split_type"] == "hetero":
        classes = torch.unique(joint_train_set.targets)
        train_subsets = []
        if N <= len(classes):
            joint_labels = joint_train_set.targets
            node_classes = torch.split(classes, int(len(classes) / N))
            for i in range(N):
                # from here: https://discuss.pytorch.org/t/tensor-indexing-with-conditions/81297/2
                locs = [lab == joint_labels for lab in node_classes[i]]
                idx_keep = torch.nonzero(torch.stack(locs).sum(0)).reshape(-1)
                train_subsets.append(
                    torch.utils.data.Subset(joint_train_set, idx_keep)
                )
        else:
            raise NameError("Hetero MNIST N > 10 not supported.")

    # Create base model
    model_conf = exp_conf["model"]
    base_model = MNISTConvNet(
        model_conf["num_filters"],
        model_conf["kernel_size"],
        model_conf["linear_width"],
    )

    # Define base loss function
    if exp_conf["loss"] == "NLL":
        base_loss = torch.nn.NLLLoss()
    else:
        raise NameError("Unknown loss function.")

    # Run each optimizer on the problem
    prob_confs = conf_dict["problem_configs"]
    for prob_key in prob_confs:
        prob_conf = prob_confs[prob_key]
        opt_conf = prob_conf["optimizer_config"]

        prob = DistMNISTProblem(
            graph,
            base_model,
            base_loss,
            train_subsets,
            val_set,
            prob_conf,
        )

        if opt_conf["alg_name"] == "cadmm":
            dopt = CADMM(prob, opt_conf)
        else:
            raise NameError("Unknown distributed opt algorithm.")

        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("Running problem: " + prob_conf["problem_name"])
        dopt.train()
        if exp_conf["writeout"]:
            prob.save_metrics(output_dir)

    # REPEAT:
    #  - Create problem
    #  - DO method initialize and train
    #  - Save metrics and additional data to output directory

    return


if __name__ == "__main__":
    yaml_pth = sys.argv[1]

    # Load the configuration file, and run the experiment
    if os.path.exists(yaml_pth):
        experiment(yaml_pth)
    else:
        raise NameError("YAML configuration file does not exist, exiting!")
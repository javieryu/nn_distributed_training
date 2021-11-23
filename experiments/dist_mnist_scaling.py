import os
import sys
from datetime import datetime
from shutil import copyfile
import copy

import yaml
from torchvision import datasets, transforms
import torch
import networkx as nx

from models.mnist_conv_nn import MNISTConvNet
from problems.dist_mnist_problem import DistMNISTProblem
from optimizers.dinno import DiNNO
from optimizers.dsgd import DSGD
from optimizers.dsgt import DSGT
from utils import graph_generation

torch.set_default_tensor_type(torch.DoubleTensor)


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

    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_dir = exp_conf["data_dir"]
    joint_train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    val_set = datasets.MNIST(data_dir, train=False, transform=transform)

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

    # Check for gpu and assign device
    if torch.cuda.is_available() and exp_conf["use_cuda"]:
        device = torch.device("cuda")
        print("Device is set to GPU")
    else:
        device = torch.device("cpu")
        print("Device is set to CPU")

    scale_conf = exp_conf["scaling"]
    if scale_conf["const"] == "fiedler":
        # Ns = list(range(scale_conf["min_scale"], scale_conf["max_scale"]))
        Ns = torch.linspace(
            scale_conf["min_N"],
            scale_conf["max_N"],
            scale_conf["num_trials"],
        )
        Ns = Ns.int().tolist()

        graphs = []
        for N in Ns:
            G = graph_generation.disk_with_fied(N, scale_conf["target_fied"])
            graphs.append(G)

    elif scale_conf["const"] == "num_nodes":
        fieds = torch.linspace(
            scale_conf["min_fied"],
            scale_conf["max_fied"],
            scale_conf["num_trials"],
        )

        graphs = []
        for fied in fieds:
            G = graph_generation.disk_with_fied(scale_conf["num_nodes"], fied)
            graphs.append(G)
    else:
        raise NameError("Unknown const factor in scaling")

    print("Graph generation successful!")

    prob_conf = conf_dict["problem"]
    for (trial, graph) in enumerate(graphs):
        N = len(graph.nodes)

        file_name = str(trial)
        prob_conf["problem_name"] = file_name

        if exp_conf["writeout"]:
            # Save the graph for future visualization
            nx.write_gpickle(
                graph, os.path.join(output_dir, file_name + ".gpickle")
            )

        train_subsets = []
        order = torch.argsort(joint_train_set.targets)
        node_data_idxs = order.chunk(N)
        for idxs in node_data_idxs:
            ds_size = len(idxs)
            train_subsets.append(
                torch.utils.data.Subset(joint_train_set, idxs)
            )

        prob = DistMNISTProblem(
            graph,
            base_model,
            base_loss,
            train_subsets,
            val_set,
            device,
            prob_conf,
        )

        opt_conf = prob_conf["optimizer_config"]

        if opt_conf["alg_name"] == "dinno":
            dopt = DiNNO(prob, device, opt_conf)
        elif opt_conf["alg_name"] == "dsgd":
            dopt = DSGD(prob, device, opt_conf)
        elif opt_conf["alg_name"] == "dsgt":
            dopt = DSGT(prob, device, opt_conf)
        else:
            raise NameError("Unknown distributed opt algorithm.")

        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("Running problem: ", trial, " / ", len(graphs))
        print("Num Nodes: ", N)
        print("DS size: ", ds_size)
        if opt_conf["profile"]:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=3, repeat=3
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(
                        output_dir, prob_conf["problem_name"] + "opt_profile"
                    )
                ),
                record_shapes=True,
                with_stack=True,
            ) as prof:
                dopt.train(profiler=prof)
        else:
            dopt.train()

        if exp_conf["writeout"]:
            prob.save_metrics(output_dir)

    return


if __name__ == "__main__":
    yaml_pth = sys.argv[1]

    # Load the configuration file, and run the experiment
    if os.path.exists(yaml_pth):
        experiment(yaml_pth)
    else:
        raise NameError("YAML configuration file does not exist, exiting!")

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
from optimizers.cadmm import CADMM
from optimizers.dsgd import DSGD
from utils import graph_generation

torch.set_default_tensor_type(torch.DoubleTensor)


def train_solo(model, loss, train_set, val_set, device, conf):
    trainloader = torch.utils.data.DataLoader(
        train_set, conf["train_batch_size"], shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        val_set, conf["val_batch_size"], shuffle=True
    )

    model = model.to(device)

    if conf["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=conf["lr"])
    elif conf["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=conf["lr"])
    elif conf["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=conf["lr"])
    else:
        raise NameError("Unknown individual optimizer.")

    for _ in range(conf["epochs"]):
        for batch in trainloader:
            opt.zero_grad()
            pd = model.forward(batch[0].to(device))
            l = loss(pd, batch[1].to(device))
            l.backward()
            opt.step()

    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        for data, labels in valloader:
            out = model.forward(data)
            val_loss += loss(out, labels).item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        val_loss /= len(valloader.dataset)
        val_acc = correct / len(valloader.dataset)

    return {"validation_loss": val_loss, "validation_accuracy": val_acc}


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
    N, graph = graph_generation.generate_from_conf(graph_conf)

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

    # Check for gpu and assign device
    if torch.cuda.is_available() and exp_conf["use_cuda"]:
        device = torch.device("cuda")
        print("Device is set to GPU")
    else:
        device = torch.device("cpu")
        print("Device is set to CPU")

    # If individual training is true then baseline
    # against training without communication.
    solo_confs = exp_conf["individual_training"]
    solo_results = {}
    if solo_confs["train_solo"]:
        print("Performing individual training ...")
        for i in range(N):
            solo_results[i] = train_solo(
                copy.deepcopy(base_model),
                base_loss,
                train_subsets[i],
                val_set,
                device,
                solo_confs,
            )

            if solo_confs["verbose"]:
                print(
                    "Node {} - Validation Acc = {:.4f}".format(
                        i, solo_results[i]["validation_accuracy"]
                    )
                )

        if exp_conf["writeout"]:
            torch.save(
                solo_results, os.path.join(output_dir, "solo_results.pt")
            )

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
            device,
            prob_conf,
        )

        if opt_conf["alg_name"] == "cadmm":
            dopt = CADMM(prob, device, opt_conf)
        elif opt_conf["alg_name"] == "dsgd":
            dopt = DSGD(prob, device, opt_conf)
        else:
            raise NameError("Unknown distributed opt algorithm.")

        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("Running problem: " + prob_conf["problem_name"])
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
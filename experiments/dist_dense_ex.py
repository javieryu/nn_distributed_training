import os
import sys
from datetime import datetime
from shutil import copyfile
import glob
import copy

import yaml
import torch
import networkx as nx
import numpy as np

sys.path.insert(0, "../models/")
sys.path.insert(0, "../problems/")
sys.path.insert(0, "../optimizers/")
sys.path.insert(0, "../utils/")
sys.path.insert(0, "../floorplans/lidar")
from fourier_nn import FourierNet
from dist_dense_problem import DistDensityProblem
from cadmm import CADMM
import graph_generation
from lidar import (
    RandomPoseLidarDataset,
    TrajectoryLidarDataset,
)

torch.set_default_tensor_type(torch.DoubleTensor)


def train_solo(model, loss, train_set, val_set, device, conf):
    """Performs normal training without any communication

    Args:
        model (torch model): the model to train
        loss (torch loss func): the loss function to train with.
        trainset (torch dataset): training dataset
        valset (torch dataset): validation dataset
        device (torch device): device to compute on (cpu or gpu)
        conf (dict): configuration dictionary (see yaml descriptions)

    Returns:
        dict: validation loss (float) after final epoch, and
            the trained models density outputs (tensor)
            evaluated on a mesh grid.
    """
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
            pd = model.forward(batch["position"].to(device))
            l = loss(pd, batch["density"].to(device))
            l.backward()
            opt.step()

    with torch.no_grad():
        vloss = 0.0
        for batch in valloader:
            pd = model.forward(batch["position"].to(device))
            vloss += loss(pd, batch["density"].to(device)).data.detach()

        X, Y = np.meshgrid(val_set.lidar.xs, val_set.lidar.ys)
        xlocs = X[::8, ::8].reshape(-1, 1)
        ylocs = Y[::8, ::8].reshape(-1, 1)
        mesh_poses = np.hstack((xlocs, ylocs))
        mesh_inputs = torch.Tensor(mesh_poses)
        mesh_inputs = mesh_inputs.to(device)

        mesh_dense = model.forward(mesh_inputs)

    return {"validation_loss": vloss, "mesh_grid_density": mesh_dense}


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

    # Create communication graph and save graph
    N, graph = graph_generation.generate_from_conf(exp_conf["graph"])

    if exp_conf["writeout"]:
        # Save the graph for future visualization
        nx.write_gpickle(graph, os.path.join(output_dir, "graph.gpickle"))

    # Load the datasets
    data_conf = exp_conf["data"]
    print("Loading the data ...")

    if data_conf["split_type"] == "random":
        train_subsets = [
            RandomPoseLidarDataset(
                os.path.join(data_conf["data_dir"], "floor_img.png"),
                data_conf["num_beams"],
                data_conf["scan_dist_scale"],
                data_conf["beam_samps"],
                data_conf["num_scans"],
                round_density=data_conf["round_density"],
            )
            for _ in range(N)
        ]
    elif data_conf["split_type"] == "trajectory":
        data_dir = data_conf["data_dir"]
        traj_pths = glob.glob(os.path.join(data_dir, "*.npy"))

        # Check that N is consistent with the number of
        # trajectories that are avaliable.
        if N > len(traj_pths):
            error_str = "Requested more nodes than there are trajectories."
            error_str += (
                "Requested {} nodes, and found {} trajectories.".format(
                    N, len(traj_pths)
                )
            )
            raise NameError(error_str)

        train_subsets = []
        for i in range(N):
            traj = np.load(traj_pths[i])
            node_set = TrajectoryLidarDataset(
                os.path.join(data_conf["data_dir"], "floor_img.png"),
                data_conf["num_beams"],
                data_conf["scan_dist_scale"],
                data_conf["beam_samps"],
                traj,
            )
            train_subsets.append(node_set)
    else:
        raise NameError(
            "Unknown data split type. Must be either (random, trajectory)."
        )

    # Print the dataset sizes
    for i in range(N):
        print("Node ", i, "train set size: ", len(train_subsets[i]))

    # Generate the validation set
    val_set = RandomPoseLidarDataset(
        os.path.join(data_conf["data_dir"], "floor_img.png"),
        data_conf["num_beams"],
        data_conf["scan_dist_scale"],
        data_conf["beam_samps"],
        data_conf["num_validation_scans"],
        round_density=data_conf["round_density"],
    )

    # Generate base model
    model_conf = exp_conf["model"]
    base_model = FourierNet(model_conf["shape"], scale=model_conf["scale"])

    # Define base loss
    if exp_conf["loss"] == "BCE":
        base_loss = torch.nn.BCELoss()
    elif exp_conf["loss"] == "MSE":
        base_loss = torch.nn.MSELoss()
    else:
        raise NameError("Unknown loss function.")

    # Check for gpu and assign device
    if torch.cuda.is_available() and exp_conf["use_cuda"]:
        device = torch.device("cuda")
        print("Device is set to GPU")
    else:
        device = torch.device("cpu")
        print("Device is set to CPU")

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
                    "Node {} - Validation loss = {:.4f}".format(
                        i, solo_results[i]["validation_loss"]
                    )
                )

        if exp_conf["writeout"]:
            torch.save(
                solo_results, os.path.join(output_dir, "solo_results.pt")
            )

    # Run each problem
    prob_confs = conf_dict["problem_configs"]

    for prob_key in prob_confs:
        prob_conf = prob_confs[prob_key]
        opt_conf = prob_conf["optimizer_config"]

        prob = DistDensityProblem(
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
        else:
            raise NameError("Unknown distributed opt algorithm.")

        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("Running problem: " + prob_conf["problem_name"])
        dopt.train()
        if exp_conf["writeout"]:
            prob.save_metrics(output_dir)


if __name__ == "__main__":
    yaml_pth = sys.argv[1]

    # Load the configuration file, and run the experiment
    if os.path.exists(yaml_pth):
        experiment(yaml_pth)
    else:
        raise NameError("YAML configuration file does not exist, exiting!")
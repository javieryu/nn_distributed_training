import os
import sys
from datetime import datetime
from shutil import copyfile
import glob

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

    # Run each problem
    prob_confs = conf_dict["problem_configs"]

    for prob_key in prob_confs:
        prob_conf = prob_confs[prob_key]
        opt_conf = prob_conf["optimizer_config"]

        prob = DistDensityProblem(
            graph, base_model, base_loss, train_subsets, val_set, prob_conf
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


if __name__ == "__main__":
    yaml_pth = sys.argv[1]

    # Load the configuration file, and run the experiment
    if os.path.exists(yaml_pth):
        experiment(yaml_pth)
    else:
        raise NameError("YAML configuration file does not exist, exiting!")
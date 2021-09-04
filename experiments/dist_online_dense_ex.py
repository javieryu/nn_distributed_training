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

from models.fourier_nn import FourierNet
from problems.dist_online_dense_problem import DistOnlineDensityProblem
from optimizers.cadmm import CADMM
from optimizers.dsgt import DSGT
from optimizers.dsgd import DSGD
from utils import graph_generation
from floorplans.lidar.lidar import (
    Lidar2D,
    OnlineTrajectoryLidarDataset,
    RandomPoseLidarDataset,
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
            pd = model.forward(batch[0].to(device))
            l = loss(pd.squeeze(), batch[1].to(device))
            l.backward()
            opt.step()

    with torch.no_grad():
        vloss = 0.0
        for batch in valloader:
            pd = model.forward(batch[0].to(device))
            vloss += loss(pd.squeeze(), batch[1].to(device)).data.detach()

        X, Y = np.meshgrid(val_set.lidar.xs, val_set.lidar.ys)
        xlocs = X[::8, ::8].reshape(-1, 1)
        ylocs = Y[::8, ::8].reshape(-1, 1)
        mesh_poses = np.hstack((xlocs, ylocs))
        mesh_inputs = torch.Tensor(mesh_poses)
        mesh_inputs = mesh_inputs.to(device)

        mesh_dense = model.forward(mesh_inputs)

    return {
        "validation_loss": vloss,
        "mesh_grid_density": mesh_dense,
        "mesh_grid": mesh_inputs,
    }


def experiment(yaml_pth):
    # load the config yaml
    with open(yaml_pth) as f:
        conf_dict = yaml.safe_load(f)

    # Seperate configuration groups
    exp_conf = conf_dict["experiment"]

    # Set seed for reproducibility
    torch.manual_seed(exp_conf["seed"])

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

    # Load the datasets, and lidar object
    data_conf = exp_conf["data"]
    print("Loading the data ...")
    img_path = os.path.join(data_conf["data_dir"], "floor_img.png")

    lidar = Lidar2D(
        img_path,
        data_conf["num_beams"],
        data_conf["beam_length"],
        data_conf["beam_samps"],
        data_conf["samp_distribution_factor"],
        data_conf["collision_samps"],
        data_conf["fine_samps"],
        border_width=data_conf["border_width"],
    )

    data_dir = data_conf["data_dir"]
    waypoint_pths = glob.glob(
        os.path.join(data_dir, data_conf["waypoint_subdir"], "*.npy")
    )
    N = len(waypoint_pths)

    # Check that N is consistent with the number of
    # trajectories that are avaliable.
    if N > len(waypoint_pths):
        error_str = "Requested more nodes than there are waypoint files."
        error_str += "Requested {} nodes, and found {} waypoint files.".format(
            N, len(waypoint_pths)
        )
        raise NameError(error_str)

    train_subsets = []
    for i in range(N):
        waypoints = np.load(waypoint_pths[i])
        node_set = OnlineTrajectoryLidarDataset(
            lidar,
            waypoints,
            data_conf["spline_res"],
            data_conf["num_scans_in_window"],
            round_density=data_conf["round_density"],
        )
        train_subsets.append(node_set)

    # Print the dataset sizes
    for i in range(N):
        print()
        print("Node ", i, "train set size: ", len(train_subsets[i]))
        print(
            "Node",
            i,
            "hd ratio: {:.4f}".format(
                (
                    torch.sum(train_subsets[i].scans[:, 2] == 1.0)
                    / train_subsets[i].scans.shape[0]
                ).data.item()
            ),
        )

    # Generate the validation set
    val_set = RandomPoseLidarDataset(
        lidar,
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
    elif exp_conf["loss"] == "L1":
        base_loss = torch.nn.L1Loss()
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

        prob = DistOnlineDensityProblem(
            base_model,
            base_loss,
            train_subsets,
            val_set,
            device,
            prob_conf,
        )

        if opt_conf["alg_name"] == "cadmm":
            dopt = CADMM(prob, device, opt_conf)
        elif opt_conf["alg_name"] == "dsgt":
            dopt = DSGT(prob, device, opt_conf)
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


if __name__ == "__main__":
    yaml_pth = sys.argv[1]

    # Load the configuration file, and run the experiment
    if os.path.exists(yaml_pth):
        experiment(yaml_pth)
    else:
        raise NameError("YAML configuration file does not exist, exiting!")

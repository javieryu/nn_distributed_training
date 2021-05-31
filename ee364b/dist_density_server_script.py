import sys
import copy
from datetime import datetime
import os

from numpy import save

sys.path.insert(0, "../models/")
sys.path.insert(0, "../lidar/")
from fourier_nn import FourierNet
from lidar import RandomPoseLidarDataset, TrajectoryLidarDataset

import torch
import networkx as nx
import numpy as np

torch.set_default_tensor_type(torch.DoubleTensor)


def primal_update(
    data_loader, data_iter, opt, model, base_loss, dual, thj, rho, lr, max_its
):
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    for k in range(max_its):
        # Load the batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        opt.zero_grad()

        yh = model.forward(batch["position"])

        # Compose CADMM Loss
        th = torch.nn.utils.parameters_to_vector(model.parameters())
        reg = torch.sum(torch.square(torch.cdist(th.reshape(1, -1), thj)))
        loss = (
            base_loss(yh, batch["density"]) + torch.dot(th, dual) + rho * reg
        )

        # Backprop and step
        loss.backward()
        opt.step()


def validate(base_loss, val_loader, model):
    val_loss = 0.0
    for batch in val_loader:
        with torch.no_grad():
            yh = model.forward(batch["position"])
            val_loss += base_loss(yh, batch["density"]).data.detach()
    return val_loss


def main():
    N = 6
    G = nx.wheel_graph(N)

    # Setup models
    shape = [2, 86, 32, 32, 32, 1]
    scale = 0.05
    base_model = FourierNet(shape, scale=scale)

    models = {i: copy.deepcopy(base_model) for i in range(N)}
    # Setup data
    data_type = "random"  # {random, trajectory}
    batch_size = 10000  # number of (x, y, z) points used to compute each grad
    validation_set_size = 300  # number of scans in the validation set

    num_beams = 20
    beam_samps = 20
    scan_dist = 0.2
    num_scans = 3000
    img_dir = "../floorplans/32_b.png"
    traj_paths = [
        "../planner/waypoint_path1.npy",
        "../planner/waypoint_path2.npy",
        "../planner/waypoint_path3.npy",
        "../planner/waypoint_path4.npy",
        "../planner/waypoint_path5.npy",
        "../planner/waypoint_path6.npy",
    ]

    train_loaders = {}  # Dictionary of dataloaders for each node
    train_iters = {}  # Dictionary of iterators for each node

    if data_type == "random":
        for i in range(N):
            node_set = RandomPoseLidarDataset(
                img_dir, num_beams, scan_dist, beam_samps, num_scans
            )
            train_loaders[i] = torch.utils.data.DataLoader(
                node_set, batch_size=batch_size, shuffle=True
            )
            train_iters[i] = iter(train_loaders[i])
    elif data_type == "trajectory":
        for i in range(N):
            traj = np.load(traj_paths[i])
            node_set = TrajectoryLidarDataset(
                img_dir, num_beams, scan_dist, beam_samps, traj
            )
            train_loaders[i] = torch.utils.data.DataLoader(
                node_set, batch_size=batch_size, shuffle=True
            )
            train_iters[i] = iter(train_loaders[i])
        print("Trajectories not yet supported")
    else:
        print("Not a supported data_type")

    val_set = RandomPoseLidarDataset(
        img_dir, num_beams, scan_dist, beam_samps, validation_set_size
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    # Setup Loss and CADMM
    primal_steps = 10
    cadmm_iterations = 2000
    eval_every = 20
    rho = 1.0
    lr = 0.005

    num_params = torch.nn.utils.parameters_to_vector(
        models[0].parameters()
    ).shape[0]
    base_loss = torch.nn.BCELoss()

    duals = {i: torch.zeros(num_params) for i in range(N)}
    obvs = torch.zeros((cadmm_iterations // eval_every + 1, N))
    opts = {
        i: torch.optim.Adam(models[i].parameters(), lr=lr) for i in range(N)
    }

    cnt_evals = 0
    for k in range(cadmm_iterations):
        ths = {
            i: (
                torch.nn.utils.parameters_to_vector(models[i].parameters())
                .clone()
                .detach()
            )
            for i in range(N)
        }

        for i in range(N):
            # Evaluate model
            if k % eval_every == 0 or k == cadmm_iterations - 1:
                obvs[cnt_evals, i] = validate(base_loss, val_loader, models[i])
                obv_str = "{:.4f}".format(obvs[cnt_evals, i].item())
                print(
                    "Iteration: ", k, " | BCELoss: ", obv_str, " | Node: ", i
                )

            # Communication
            neighs = list(G.neighbors(i))
            thj = torch.stack([ths[j] for j in neighs])

            # Update the dual var
            duals[i] += rho * torch.sum(ths[i] - thj, dim=0)

            # Primal Update
            # (data_loader, data_iter, model, base_loss, dual, thj, rho, lr, max_its)
            primal_update(
                train_loaders[i],
                train_iters[i],
                opts[i],
                models[i],
                base_loss,
                duals[i],
                thj,
                rho,
                lr,
                primal_steps,
            )

        if k % eval_every == 0:
            cnt_evals += 1

    save_dir = (
        "outputs/dense_"
        + data_type
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(save_dir)

    torch.save(
        {"shape": shape, "scale": scale, "objvals": obvs},
        save_dir + "/misc.pt",
    )
    model_sds = {i: models[i].state_dict() for i in range(N)}
    torch.save(model_sds, save_dir + "/allmodels.pt")


if __name__ == "__main__":
    main()

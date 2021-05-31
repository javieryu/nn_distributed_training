import os
from datetime import datetime
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 1)
        # self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def primal_update(
    data_loader, data_iter, model, base_loss, dual, thj, rho, lr, max_its
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(max_its):
        # Load the batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, y = next(data_iter)

        opt.zero_grad()

        yh = model.forward(x)

        # Compose CADMM Loss
        th = torch.nn.utils.parameters_to_vector(model.parameters())
        reg = torch.sum(torch.square(torch.cdist(th.reshape(1, -1), thj)))
        loss = base_loss(yh, y) + torch.dot(th, dual) + rho * reg

        # Backprop and step
        loss.backward()
        opt.step()


def validate(base_loss, val_loader, model):
    with torch.no_grad():
        loss = 0.0
        correct = 0
        for x, y in val_loader:
            yh = model.forward(x)
            loss += base_loss(yh, y).item()
            pred = yh.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
        avg_loss = loss / len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)
        return avg_loss, acc


def train_ind(model, train_loader, val_loader):
    lr = 0.01
    epochs = 5

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        for idx, (data, labels) in enumerate(train_loader):
            opt.zero_grad()
            out = model.forward(data)
            loss = F.nll_loss(out, labels)
            loss.backward()
            opt.step()

            if idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        ep,
                        idx * len(data),
                        len(train_loader.dataset),
                        100.0 * idx / len(train_loader),
                        loss.item(),
                    )
                )

    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        for data, labels in val_loader:
            out = model.forward(data)
            test_loss += F.nll_loss(out, labels).item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(val_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(val_loader.dataset),
                100.0 * correct / len(val_loader.dataset),
            )
        )

    return test_loss, correct


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    val_set = datasets.MNIST("../data", train=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000)

    N = 10
    G = nx.wheel_graph(N)

    base_model = Net()
    models = {i: copy.deepcopy(base_model) for i in range(N)}

    # Setup Data
    data_split_type = "hetero"
    labels = train_set.targets
    train_idxs = np.arange(len(labels))
    batch_size = 64

    train_loaders = {}
    train_iters = {}

    if data_split_type == "hetero":
        print("Data Split Type: Heterogenous")
        for i in range(N):
            idx_to_keep = labels == i
            node_subset = torch.utils.data.Subset(
                train_set, train_idxs[idx_to_keep]
            )
            train_loaders[i] = torch.utils.data.DataLoader(
                node_subset, batch_size=batch_size, shuffle=True
            )
            train_iters[i] = iter(train_loaders[i])
    else:
        print("Data Split Type: Uniform Random")
        num_per = len(labels) / N
        splits = [int(num_per) for _ in range(N)]
        uniform_sets = torch.utils.data.random_split(train_set, splits)
        for i in range(N):
            train_loaders[i] = torch.utils.data.DataLoader(
                uniform_sets[i], batch_size=batch_size, shuffle=True
            )
            train_iters[i] = iter(train_loaders[i])

    # Save Directory
    save_dir = (
        "outputs/class_"
        + data_split_type
        + "_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(save_dir)

    # Solo Problems
    solo_results = torch.zeros((N, 2))

    for i in range(N):
        solo_results[i, 0], solo_results[i, 1] = train_ind(
            copy.deepcopy(base_model), train_loaders[i], val_loader
        )

    torch.save(solo_results, save_dir + "/soloresults.pt")

    # Setup Loss and CADMM
    primal_steps = 5
    cadmm_iterations = 2000
    eval_every = 20
    rho = 1.0
    lr = 0.005

    num_params = torch.nn.utils.parameters_to_vector(
        models[0].parameters()
    ).shape[0]

    duals = {i: torch.zeros(num_params) for i in range(N)}
    obvs = torch.zeros((cadmm_iterations // eval_every + 1, N))
    accs = torch.zeros((cadmm_iterations // eval_every + 1, N))

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
                models[i],
                torch.nn.NLLLoss(),
                duals[i],
                thj,
                rho,
                lr,
                primal_steps,
            )

            # Evaluate model on all classes
            if k % eval_every == 0 or k == cadmm_iterations - 1:
                obvs[cnt_evals, i], accs[cnt_evals, i] = validate(
                    torch.nn.NLLLoss(), val_loader, models[i]
                )
                obv_str = "{:.4f}".format(obvs[cnt_evals, i].item())
                acc_str = "{:.4f}".format(accs[cnt_evals, i].item())
                print(
                    "Iteration: ",
                    k,
                    " | NLLLoss: ",
                    obv_str,
                    " | Acc: ",
                    acc_str,
                    " | Node: ",
                    i,
                )

        if k % eval_every == 0:
            cnt_evals += 1

    torch.save(
        {"obj_vals": obvs, "accuracy": accs},
        save_dir + "/dist_class_results.pt",
    )


if __name__ == "__main__":
    main()

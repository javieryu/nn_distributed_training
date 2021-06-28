import os
import sys
from datetime import datetime
from shutil import copyfile

import yaml
from torchvision import datasets, transforms
import torch
import networkx as nx

from models import mnist_conv_nn


def experiment(yaml_pth):
    # load the config yaml
    with open(yaml_pth) as f:
        conf_dict = yaml.safe_load(f)

    # Seperate configuration groups
    exp_conf = conf_dict["experiment"]
    opt_confs = conf_dict["optimizer_configs"]

    # Create the output directory
    output_metadir = exp_conf["output_metadir"]
    if not os.path.exists(output_metadir):
        os.mkdir(output_metadir)

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = os.path.join(
        output_metadir, time_now + "_" + exp_conf["name"]
    )

    os.mkdir(output_dir)
    exp_conf["output_dir"] = output_dir  # probably bad practice

    # Save a copy of the conf to the output directory
    copyfile(yaml_pth, os.path.join(output_dir, time_now + ".yaml"))

    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_dir = exp_conf["data_dir"]
    train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    val_set = datasets.MNIST(data_dir, train=False, transform=transform)

    # Create base model
    model_conf = exp_conf["model"]
    base_model = mnist_conv_nn.MNISTConvNet(
        model_conf["num_filters"],
        model_conf["kernel_size"],
        model_conf["linear_width"],
    )

    # Create communication graph
    # Define base loss function

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
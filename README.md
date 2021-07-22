# nn_distributed_training

Collective training of neural networks on distributed datasets.

## Setting up the venv

- Install ``pip`` and ``virtualenv`` if you don't have them already.

- Clone the repository and navigate to the folder in a terminal.

- Run ``virtualenv venv`` to make a new virtual environment for the project in the repo folder (the ``./venv/`` folder is git ignored so it won't push those files to the repo).

- Activate the virtual environment with: ``source venv/bin/activate``.

- Install the requirements with: ``pip install -r requirements.txt``.

- If you've installed new packages and want to add them to the ``requirements.txt`` just do: ``pip freeze > requirements.txt``.

## Directory organization and contents

- ``data/`` - A git ignored folder that I use to store relevant datasets 

- ``ee364b/`` - Old code from the Convex II class that we can leave around until the refactor is over.

- ``experiments/`` - Where "meta" scripts and configurations for running distributed learning experiments are located.
  - Additionally, ``experiments/outputs/`` is the default location for writing out the results and storing configurations from different experiments.

- ``floorplans/`` - Contains images and scripts relevant to pre-processing of CubiCasa5k images for simulating lidar enabled robots.

- ``lidar/`` - Contains a module that is used to extract simulated lidar scans from floorplan images (CubiCasa5k) and custom PyTorch Datasets and Dataloaders for integration with the distributed learning pipeline. Additionally, contains a notebook with a simple example showing how to use the lidar module.

- ``optimizers/`` - Directory for storing distributed learning algorithm modules. These modules should be readable, and agnostic to the distributed learning problem (more on that later).
  - Currently only CADMM is implemented, but hoping to add Prox-PDA and Choco-SGD.

- ``planner/`` - Contains a module with a simple waypoint based A* planner that is to be used in conjunction with the ``lidar`` module to simulated scans from physically realizable trajectories.

- ``problems/`` - Contains modules that work in conjunction with the meta-scripts in the ``experiments/`` folder. These problems modules are used to store the local models for a specific distributed learning problem, and implement problem (ex: MNIST classification) specific functions like evaluation metrics and computing local losses.

- ``visualization/`` - A directory for maintaining various notebooks for visualizing the results from experiments so that any figures are reproducible. Specifically, notebooks in this folder pull data from the ``experiments/outputs/`` folder and create useful visualizations.

## Architecture

I'm not a software engineer, but here is roughly the system architecture/hierarchy that I have been using.

- **Meta-level (experiments scripts):** These scripts are used to organize experiments that consist of comparing multiple distributed optimization runs against one another in a reproducible way and with a single configuration file. For instance, comparing DSGD to CADMM on a MNIST problem, or multiple configurations of CADMM on the same MNIST configuration.

- **Problem-level (problem and optimizer modules):** This lower level has two parts that work in tandem the problem module and the optimizer module. The idea here is that we use the problem module to take care of domain specific functions (ex: functions like computing the accuracy of a model on the MNIST validation set), and then the optimizer module that coordinates the distributed learning problem. This seperation of the optimizer and the problem makes the optimizer more readable because they aren't cluttered with domain specific computation, and in turn allows the optimizer to be used for any type of problem (ex: implicit density or MNIST).

The idea here is that each experiment script runs multiple (problem, optimizer) pairs, and then stores the results from each in a single place. This keeps a number of variables consistent:

- By never modifying the "base model" the initial weights for every node in every (problem, optimizer) pair of the experiment are exactly the same.

- We've seen that the distribution of data at each node has a strong influence on the convergence rates of these methods, and this can be effected by randomization of the data sets. The experiments script keeps the data subsets for each node the same across each (problem, optimizer) pair. 

## Experiment configuration YAML parameters



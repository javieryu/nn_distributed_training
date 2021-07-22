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

- **Problem-level (problem and optimizer modules):** This lower level has two parts that work in tandem the problem module and the optimizer module. The idea here is that we use the problem module to take care of domain specific functions (ex: functions like computing the accuracy of a model on the MNIST validation set), and then the optimizer module that coordinates the distributed learning problem. This separation of the optimizer and the problem makes the optimizer more readable because they aren't cluttered with domain specific computation, and in turn allows the optimizer to be used for any type of problem (ex: implicit density or MNIST).

The idea here is that each experiment script runs multiple (problem, optimizer) pairs, and then stores the results from each in a single place. This keeps a number of variables consistent:

- By never modifying the "base model" the initial weights for every node in every (problem, optimizer) pair of the experiment are exactly the same.

- We've seen that the distribution of data at each node has a strong influence on the convergence rates of these methods, and this can be effected by randomization of the data sets. The experiments script keeps the data subsets for each node the same across each (problem, optimizer) pair.

## Experiment configuration YAML parameters

The an experiment is run with a YAML file that specifies the configuration for the various parameters of the experiment. Configuration YAMLs are specific to each domain, so one that is written for MNIST will only work for MNIST.

To run an experiment:

```bash
# Starting in repo root, and assuming you have
# the virtualenv configured
> source venv/bin/activate
> cd experiments/
> python [experiment script name] [path to yaml file]

# For example:
> python dist_mnist_ex.py dist_mnist_template.yaml
```

For each experiment the YAML file that was used to run it is saved to its output directory in case you want to rerun that experiment with those configurations. That said, we will probably make changes that will deprecate old YAMLs so its useful to make not of which version of the codebase that you run the experiment on for maximal reproducibility.

### MNIST YAML configurations

The YAML file located at ``experiments/dist_mnist_template.yaml`` provides a template for running different problems on the distributed MNIST classifier problem. Below is a breakdown of the supported parameters in the MNIST YAML.

The configuration is broken into two main parts experiment configuration, and problem configuration.

- Experiment configuration: determines the "meta" parameters of the experiment. For example the data distributions at each node. NOTE: while it would be cool to test across some of these experiment parameters right now that requires running two separate experiments and then comparing the outputs.

- Problem configuration: specifies problem specific parameters and the parameters of the specific optimizer to be used with that problem. Each experiment might have several problem configurations which will be processed one after another.

#### **Experiment Configurations**

```yaml
experiment:
  name: "dist_cnn_cadmm_scaled_v_const"
  data_dir: "../data/"
  output_metadir: "./outputs/"
  writeout: true
  data_split_type: "hetero"
  loss: "NLL"
  graph:
    num_nodes: 5
    type: "wheel"
    p: 0.6
    gen_attempts: 10
  model:
    num_filters: 3
    kernel_size: 5
    linear_width: 64
...
```

- ``name: [string]`` - The string that is used to create the output directory that is going to be used for this experiment. The directories are automatically generated, and have the format ``[output_metadir]/[date and time of execution]_[experiment name]/``.

- ``data_dir: [path]`` - Designates the location of the MNIST dataset to be used, and in the case that this directory is empty then the MNIST dataset is downloaded using the TorchVision API.

- ``output_metadir: [path]`` - For use in case you want to change the directory that the results are written out to.

- ``writeout: [bool]`` - Used for testing/debugging, if this is false then nothing from this experiment will be saved.

- ``data_split_type: [hetero, random]`` - Determines how the MNIST data is divided between the nodes. Option ``hetero`` corresponds to digit classes are divided between nodes (ex: node 1 only has examples of digits two and four), and option ``random`` corresponds to an equal and random division of the training examples between the nodes.

- ``loss: [NLL]`` -  The loss function used for the classifier. For now only the negative log likelihood ``NLL`` is supported for MNIST.

- ``graph`` - Parameters corresponding specifically to the communication graph.

  - ``num_nodes: [int]``- The number of nodes in the distributed optimization. NOTE: If the data split type is ``hetero`` then more than 10 nodes is currently not supported.

  - ``type: [wheel, random]`` - The type of graph generated for the experiment. ``wheel`` is a wheel graph (deterministic), and ``random`` is a Erdos-Renyi graph which uses the following two parameters in its generation.

  - ``p: [0.0 - 1.0]`` - If the graph type is ``random`` then ``p`` is the probability that any two nodes will be connected.

  - ``gen_attempts: [int]`` - If the graph type is ``random`` then it can be the case that a non-connected graph is generated. In that case the experiment script will attempt to generated a new connected graph, and if after ``gen_attempts`` number of attempts there is no connected graph then the script will error.

- ``model:`` - Parameters associated with the CNN used for this experiment. Currently the architecture that is used is one ReLU activated convolutional layer, a max-pool2d layer, and then two fully connected layers.

  - ``num_filters: [int]`` - Number of filters used in the convolution layer.

  - ``kernal_width: [int]`` - Width of the convolution filters.

  - ``linear_width: [int]`` - Width of the second linear layer (the width of the first is determined by the two parameters above).

#### **Problem Configurations**

```yaml
...
problem_configs:
  problem1:
    problem_name: "vanilla_cadmm"
    train_batch_size: 64
    val_batch_size: 128
    evaluate_frequency: 20
    verbose_evals: True
    metrics:
      - "forward_pass_count"
      - "validation_loss"
      - "consensus_error"
      - "top1_accuracy"
      - "current_epoch"
    optimizer_config:
      alg_name: "cadmm"
      rho_init: 1.0
      rho_scaling: 1.0
      outer_iterations: 30
      primal_iterations: 5
      primal_optimizer: "adam"
      primal_lr: 0.005
...
```

- The section below the header ``problem_configs`` is broken into "sub" configurations which can be an unlimited number of (problem, optimizer) pairs. The subsection header name (in this case ``problem1``) doesn't matter it's just used to delineate between the different problems for this particular experiment. In general, the parameters that actually make each problem different are generally the optimizer parameters, but later this could be expanded.

- ``problem_name: [string]`` - Used to uniquely identify the generated results file from this problem. The results file will have path ``[generated output dir]/[problem_name]_results.pt``.

- ``train_batch_size: [int]`` - The batch size that each model will use during training to compute the current loss during training.

- ``val_batch_size: [int]`` - The batch size that is used when validating a model (usually we just want this to be as large as possible without running out of GPU memory or RAM).

- ``evaluate_frequency: [int]`` - The number of communication rounds between evaluations of the specified metrics (see below). This is to cut down on the number of times we evaluate the models which can get pretty expensive.

- ``verbose_evals: [bool]`` - Whether or not to print the ranges of the metrics during after each evaluation.

- ``metrics: [list]`` - What metrics to compute during the evaluations. The values of each of these metrics after each evaluation are written out to the results file that is generated at the end of the training. Currently supported metrics:

  - ``forward_pass_count`` - The number of forward passes that each node has performed.

  - ``validation_loss`` - The performance of each model on the validation set.

  - ``consensus_error`` - Pairwise distances between the weights of all nodes. This returns a distance matrix in the results but the printed values are the min and max of the average distances of the nodes.

  - ``top1_accuracy`` - The accuracy of each of the models on the validation set.

  - ``current_epoch`` - This is useful when the amount of data at each node is not balanced, and because of the fixed batch size some nodes will loop through their entire data set faster than others.

- ``optimizer_configs`` - These are the parameters that specify the parameters of the distributed optimization algorithm used for this problem. **NOTE**: Parameters following ``alg_name`` will be different for each algorithm so I've broken up the following descriptions accordingly.

  - ``alg_name: [cadmm]`` - Currently only CADMM is supported, but hoping to later add Prox-PDA, DSGD, and NEXT.

  - ``cadmm`` parameters:

    - ``rho_init: [float]`` - The initial ADMM penalty parameter.

    - ``rho_scaling: [float >= 1.0]`` - After each communication round the penalty parameter for all nodes is scaled by multiplication of this amount.

    - ``outer_iterations: [int]`` - The number of communication rounds performed in total for this problem.

    - ``primal_iterations: [int]`` - The number of steps that the inner optimizer takes during each ADMM primal update.

    - ``primal_optimizer: [adam, adamw, sgd]`` - Which PyTorch optimizer to use to perform the primal update.

    - ``primal_lr: [float]`` - The learning rate of the primal optimizer.

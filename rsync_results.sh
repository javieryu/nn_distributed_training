#!/bin/bash
LOCALDIR=/home/javier/workspace/nn_distributed_training
REMOTE=homer
REMOTEDIR=/home/javier/workspace/nn_distributed_training/results_scaling
OPTS="-avzhe ssh --progress"
rsync $OPTS $REMOTE:$REMOTEDIR $LOCALDIR
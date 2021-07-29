#!/bin/bash
LOCALDIR=/home/javier/workspace/nn_distributed_training
REMOTE=javieryu@bespin.stanford.edu
REMOTEDIR=/home/javieryu/workspace/nn_distributed_training/results
OPTS="-avzhe ssh --progress"
rsync $OPTS $REMOTE:$REMOTEDIR $LOCALDIR
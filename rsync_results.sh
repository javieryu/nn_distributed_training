#!/bin/bash
LOCALDIR=/home/javieryu/workspace/nn_distributed_training
REMOTE=javieryu@bespin.stanford.edu
REMOTEDIR=~/workspace/nn_distributed_training/results
OPTS="--dry-run -avzhe ssh --progress"
rsync $OPTS $REMOTE:$REMOTEDIR $LOCALDIR
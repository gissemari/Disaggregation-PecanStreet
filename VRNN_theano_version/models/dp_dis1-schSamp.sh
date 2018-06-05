#!/bin/bash
#SBATCH -J dp-dis1sched
echo "starting own vrnn"
export PYTHONPATH=/data/home/gbejara1/Documents/Research/Disaggregation/Dataport:/data/home/gbejara1/Documents/Research/Disaggregation/UK-DALE/VRNN_theano_version/models
source activate latentVar
THEANO_FLAGS="gcc.cxxflags='-march=core2'" python vrnn_diss1-priorXt-schedSamp.py
echo "ending"

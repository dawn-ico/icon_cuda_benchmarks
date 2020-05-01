#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/atlas_install/release/cpu/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/eckit_install/lib

srun -A s83 -C gpu -p debug -u ./out $1

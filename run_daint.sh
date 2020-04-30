#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/atlas_install/release/cpu/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/eckit_install/lib

srun --account c14 -C gpu --partition debug --gres=gpu:1 -u ./out 340

#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/mroeth/tsa/spack/spack-install/tsa/atlas/develop/gcc/2bgfzwgc7pxmx32wojgdthepqli7gng7/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/mroeth/tsa/spack/spack-install/tsa/eckit/master/gcc/kcvtp4xpjy7fbk7oz73o45a55jyd2xgv/lib/

srun -A s83 -C gpu -p debug -u --gres=gpu:1 ./out $1
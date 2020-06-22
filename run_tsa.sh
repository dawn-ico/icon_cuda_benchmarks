#!/bin/bash

module load gcc/8.3.0
module load cuda/10.1.243

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/cosuna/tsa/spack/spack-install/tsa/atlas/develop/gcc/2bgfzwgc7pxmx32wojgdthepqli7gng7/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/cosuna/tsa/spack/spack-install/tsa/eckit/master/gcc/kcvtp4xpjy7fbk7oz73o45a55jyd2xgv/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/arolla/UES/jenkins/RH7.6/gnu/19.2/easybuild/software/netCDF-C++/4.3.0-fosscuda-2019b/lib

srun -C gpu --partition debug --gres=gpu:1 -u ./out $@

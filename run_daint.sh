#!/bin/bash -l
#
#SBATCH --job-name="diamond_stencil_test"
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --account c14
#SBATCH --partition debug
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --output=diamond_stencil_test.%j.output.log
#SBATCH --error=diamond_stencil_test.%j.error.log

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/atlas_install/release/cpu/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/eckit_install/lib

srun -C gpu ./out 64

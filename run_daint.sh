#!/bin/bash

module load gcc/8.3.0
module load cudatoolkit/10.1.105_3.27-7.0.1.1_4.1__ga311ce7

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/snx3000/mroeth/daint/spack/spack-install/daint/atlas/develop/gcc/shastqtqvp3umrxnbbi24qo22dgg4iar/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/snx3000/mroeth/daint/spack/spack-install/daint/eckit/master/gcc/smpn5fkts67iyfyseaspzhcnzfzvtlxk/lib

srun --account c14 -C gpu --partition debug --gres=gpu:1 -u ./out $1

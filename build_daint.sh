#!/bin/bash

module load gcc/8.3.0
module load cudatoolkit/10.1.105_3.27-7.0.1.1_4.1__ga311ce7 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/snx3000/mroeth/atlas/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/snx3000/mroeth/eckit/install/lib 

atlasInc=/scratch/snx3000/mroeth/atlas/install/include
eckitInc=/scratch/snx3000/mroeth/eckit/install/include

atlasLib=/scratch/snx3000/mroeth/atlas/install/lib
eckitLib=/scratch/snx3000/mroeth/eckit/install/lib

atlasUtilsInc=/scratch/snx3000/mroeth/atlas_utils/utils/

cudaInc=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/include
cudaLib=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/lib64

nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc

g++ -o out cuda_driver.cpp cuda_stencil.o $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp $atlasUtilsInc/GenerateRectAtlasMesh.cpp -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl




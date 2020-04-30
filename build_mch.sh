#!/bin/bash

atlasInc=/home/mroeth/workspace/atlas/build/install/include
eckitInc=/home/mroeth/workspace/eckit/build/install/include

atlasLib=/home/mroeth/workspace/atlas/build/install/lib
eckitLib=/home/mroeth/workspace/eckit/build/install/lib

atlasUtilsInc=/home/mroeth/workspace/atlas_utils

cudaInc=/usr/local/cuda-10.1/include
cudaLib=/usr/local/cuda-10.1/lib64/

nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -g -arch=sm_70 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc

g++-9 -o out cuda_driver.cpp cuda_stencil.o $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp -std=c++17 -g -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl
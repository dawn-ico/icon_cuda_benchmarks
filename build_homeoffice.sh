#!/bin/bash

atlasInc=/home/matthias/workspace/atlas/build/install/include
eckitInc=/home/matthias/workspace/eckit/build/install/include

atlasLib=/home/matthias/workspace/atlas/build/install/lib
eckitLib=/home/matthias/workspace/eckit/build/install/lib

atlasUtilsInc=/home/matthias/workspace/AtlasUtils/utils
atlasIOInc=/home/matthias/workspace/AtlasUtils/stencils

cudaInc=/usr/local/cuda-10.2/include
cudaLib=/usr/local/cuda-10.2/lib64/

#nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc
nvcc -c -o diamond_stencil.o diamond_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc

#g++-9 -o out cuda_driver.cpp cuda_stencil.o $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp $atlasUtilsInc/GenerateRectAtlasMesh.cpp -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl
g++-9 -o out diamond_driver.cpp diamond_stencil.o $atlasIOInc/io/atlasIO.cpp $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp $atlasUtilsInc/GenerateRectAtlasMesh.cpp -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -I$atlasIOInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl
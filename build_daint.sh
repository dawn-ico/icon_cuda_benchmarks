#!/bin/bash

module load gcc/8.3.0
module load cudatoolkit/10.1.105_3.27-7.0.1.1_4.1__ga311ce7 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/atlas_install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/c14/install/daint/eckit_install

atlasInc=/project/c14/install/daint/atlas_install/release/cpu/include
eckitInc=/project/c14/install/daint/eckit_install/include

atlasLib=/project/c14/install/daint/atlas_install/release/cpu/lib
eckitLib=/project/c14/install/daint/eckit_install/lib

atlasUtilsInc=/scratch/snx3000/mroeth/atlas_utilities/utils
atlasIOInc=/scratch/snx3000/mroeth/atlas_utilities/stencils

cudaInc=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/include
cudaLib=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/lib64

#nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc
nvcc -c -o diamond_stencil.o diamond_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc

g++ -o out diamond_driver.cpp diamond_stencil.o $atlasIOInc/io/atlasIO.cpp $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp $atlasUtilsInc/GenerateRectAtlasMesh.cpp -std=c++17 -fsanitize=address -I$atlasUtilsInc -I$atlasIOInc -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl -lasan
#g++ -o out cuda_driver.cpp cuda_stencil.o $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp $atlasUtilsInc/GenerateRectAtlasMesh.cpp -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl




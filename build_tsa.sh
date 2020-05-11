#!/bin/bash

module load gcc/8.3.0
module load cuda/10.1.243

atlasDir=/users/mroeth/tsa/spack/spack-install/tsa/atlas/develop/gcc/2bgfzwgc7pxmx32wojgdthepqli7gng7/
eckitDir=/users/mroeth/tsa/spack/spack-install/tsa/eckit/master/gcc/kcvtp4xpjy7fbk7oz73o45a55jyd2xgv/
atlasUtilsDir=/users/mroeth/tsa/spack/spack-install/tsa/atlas_utilities/master/gcc/lrtmaiwn7mhdhkyxpc7pqbieur2kaxtb

atlasInc=${atlasDir}/include
eckitInc=${eckitDir}/include

atlasLib=${atlasDir}/lib
eckitLib=${eckitDir}/lib

atlasUtilsInc=${atlasUtilsDir}/include
atlasUtilsLib=${atlasUtilsDir}/lib

cudaInc=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/include
cudaLib=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/lib64

#nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc
#nvcc -c -o diamond_stencil.o diamond_stencil.cu -std=c++14 -arch=sm_70 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc

nvcc -o out diamond_driver.cu diamond_stencil.o -std=c++14 -arch=sm_70 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasUtilsLib -latlasIOLib  -latlasUtilsLib -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -lrt -ldl
#g++ -o out cuda_driver.cpp cuda_stencil.o $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp $atlasUtilsInc/GenerateRectAtlasMesh.cpp -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl




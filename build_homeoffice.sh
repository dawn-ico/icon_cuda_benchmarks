#!/bin/bash

atlasDir=/home/matthias/tsa/spack/spack-install/tsa/atlas/develop/gcc/uwa37lxd2hxa6maqdxzv4zzqabq5pt6s/
eckitDir=/home/matthias/tsa/spack/spack-install/tsa/eckit/master/gcc/y3m435mbcipnbeucucgtqifddmwq2udg/
atlasUtilsDir=/home/matthias/tsa/spack/spack-install/tsa/atlas_utilities/master/gcc/hs4paexsvlaydgaakv5isvyojzre4pze/

atlasInc=${atlasDir}/include
eckitInc=${eckitDir}/include

atlasLib=${atlasDir}/lib
eckitLib=${eckitDir}/lib

atlasUtilsInc=${atlasUtilsDir}/include
atlasUtilsLib=${atlasUtilsDir}/lib

cudaInc=/usr/local/cuda-10.2/include
cudaLib=/usr/local/cuda-10.2/lib64/

#nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc
# nvcc -c -o diamond_stencil.o diamond_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc

#g++-9 -o out cuda_driver.cpp cuda_stencil.o $atlasUtilsInc/AtlasCartesianWrapper.cpp $atlasUtilsInc/AtlasExtractSubmesh.cpp $atlasUtilsInc/GenerateRectAtlasMesh.cpp -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl
nvcc -o out diamond_driver.cu -std=c++14 -arch=sm_60 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasUtilsLib -latlasIOLib  -latlasUtilsLib -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -lrt -ldl
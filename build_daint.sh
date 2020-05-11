#!/bin/bash

module load gcc/8.3.0
module load cudatoolkit/10.1.105_3.27-7.0.1.1_4.1__ga311ce7

atlasDir=/scratch/snx3000/mroeth/daint/spack/spack-install/daint/atlas/develop/gcc/shastqtqvp3umrxnbbi24qo22dgg4iar/
eckitDir=/scratch/snx3000/mroeth/daint/spack/spack-install/daint/eckit/master/gcc/smpn5fkts67iyfyseaspzhcnzfzvtlxk/
atlasUtilsDir=/scratch/snx3000/mroeth/daint/spack/spack-install/daint/atlas_utilities/master/gcc/xfv37ptrvuf657srvivvjv472ijflrd3

atlasInc=${atlasDir}/include
eckitInc=${eckitDir}/include

atlasLib=${atlasDir}/lib
eckitLib=${eckitDir}/lib

atlasUtilsInc=${atlasUtilsDir}/include
atlasUtilsLib=${atlasUtilsDir}/lib

cudaInc=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/include
cudaLib=/opt/nvidia/cudatoolkit10/10.1.105_3.27-7.0.1.1_4.1__ga311ce7/lib64

#nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc
nvcc -c -o diamond_stencil.o diamond_stencil.cu -std=c++14 -arch=sm_60 -I$atlasInc -I$eckitInc -L$atlasLib -latlas -L$eckitLib -leckit -I$atlasUtilsInc

echo "g++ -o out diamond_driver.cpp diamond_stencil.o  -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasUtilsLib -latlasIOLib  -latlasUtilsLib -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl"

g++ -o out diamond_driver.cpp diamond_stencil.o  -std=c++17 -I$atlasUtilsInc -I$atlasInc -I$eckitInc -L$atlasUtilsLib -latlasIOLib  -latlasUtilsLib -L$atlasLib -latlas -L$eckitLib -leckit -I$cudaInc -L$cudaLib -lcudart_static -pthread -lrt -ldl
#!/bin/bash
nvcc -c -o cuda_stencil.o cuda_stencil.cu -std=c++14 -arch=sm_60 -I/home/matthias/workspace/atlas/build/install/include -I/home/matthias/workspace/eckit/build/install/include -L/home/matthias/workspace/atlas/build/install/lib -latlas -L/home/matthias/workspace/eckit/build/install/lib -leckit -I/home/matthias/workspace/AtlasUtils/

g++-9 -o out cuda_driver.cpp cuda_stencil.o /home/matthias/workspace/AtlasUtils/AtlasCartesianWrapper.cpp /home/matthias/workspace/AtlasUtils/AtlasExtractSubmesh.cpp -std=c++17 -I/home/matthias/workspace/AtlasUtils/ -I/home/matthias/workspace/atlas/build/install/include -I/home/matthias/workspace/eckit/build/install/include -L/home/matthias/workspace/atlas/build/install/lib -latlas -L/home/matthias/workspace/eckit/build/install/lib -leckit -I/usr/local/cuda-10.2/include -L /usr/local/cuda-10.2/lib64/ -lcudart_static -pthread -lrt -ldl




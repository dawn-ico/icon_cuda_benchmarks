//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "cuda_stencil.h"

#include <atlas/mesh/Elements.h>
#include <atlas/mesh/HybridElements.h>
#include <atlas/mesh/Nodes.h>
#include <atlas/util/CoordinateEnums.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
#define EDGES_PER_NODE 6
#define EDGES_PER_CELL 3
#define CELLS_PER_EDGE 2
#define NODES_PER_EDGE 2
#define BLOCK_SIZE 128
__global__ void computeRot(const int* __restrict__ nodeToEdge, const double* __restrict__ vecE,
                           const double* __restrict__ geofacRot, int numNodes, double* rotVec) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numNodes) {
    return;
  }
  {
    double lhs = 0.;                                            // init
    for(int nbhIter = 0; nbhIter < EDGES_PER_NODE; nbhIter++) { // reduceEdgeToVertex
      int nbhIdx = __ldg(&nodeToEdge[pidx * EDGES_PER_NODE + nbhIter]);
      lhs += __ldg(&vecE[nbhIdx]) * __ldg(&geofacRot[pidx * EDGES_PER_NODE + nbhIter]);
    }
    rotVec[pidx] += lhs;
  }
}
__global__ void computeDiv(const int* __restrict__ cellToEdge, const double* __restrict__ vecE,
                           const double* __restrict__ geofacDiv, int numCells, double* divVec) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numCells) {
    return;
  }
  {
    double lhs = 0.;                                            // init
    for(int nbhIter = 0; nbhIter < EDGES_PER_CELL; nbhIter++) { // reduceEdgeToCell
      int nbhIdx = __ldg(&cellToEdge[pidx * EDGES_PER_CELL + nbhIter]);
      lhs += __ldg(&vecE[nbhIdx]) * __ldg(&geofacDiv[pidx * EDGES_PER_CELL + nbhIter]);
    }
    divVec[pidx] += lhs;
  }
}
__global__ void computeLapl(const int* __restrict__ edgeToNode, const double* __restrict__ rotVec,
                            const double* __restrict__ primal_edge_length,
                            const double* __restrict__ tangent_orientation,
                            const int* __restrict__ edgeToCell, const double* __restrict__ divVec,
                            const double* __restrict__ dual_edge_length, int numEdges,
                            double* nabla2vec) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  {
    double lhs = 0.;
    double weights[2] = {1, -1}; // compile time literals, can be generated that way
    for(int nbhIter = 0; nbhIter < NODES_PER_EDGE; nbhIter++) { // reduceNodeToEdge
      int nbhIdx = __ldg(&edgeToNode[pidx * NODES_PER_EDGE + nbhIter]);
      lhs += __ldg(&rotVec[nbhIdx]) * weights[nbhIter];
    }
    nabla2vec[pidx] += lhs;
  }
  nabla2vec[pidx] =
      nabla2vec[pidx] / __ldg(&primal_edge_length[pidx]) * __ldg(&tangent_orientation[pidx]);
  {
    double lhs = 0.;
    double weights[2] = {1, -1}; // compile time literals, can be generated that way
    for(int nbhIter = 0; nbhIter < CELLS_PER_EDGE; nbhIter++) { // reduceCellToEdge
      int nbhIdx = __ldg(&edgeToCell[pidx * CELLS_PER_EDGE + nbhIter]);
      lhs += __ldg(&divVec[nbhIdx]) * weights[nbhIter];
    }
    nabla2vec[pidx] += lhs;
  }
  nabla2vec[pidx] = nabla2vec[pidx] / __ldg(&dual_edge_length[pidx]);
}
} // namespace

template <typename ConnectivityT>
void copyNeighborTable(const ConnectivityT& conn, int numElements, int numNbhPerElement,
                       int* target) {
  {
    std::vector<int> nodeToCellHost;
    for(int nodeIdx = 0; nodeIdx < numElements; nodeIdx++) {
      for(int nbhIdx = 0; nbhIdx < conn.cols(nodeIdx); nbhIdx++) {
        nodeToCellHost.push_back(conn(nodeIdx, nbhIdx));
      }
    }
    assert(nodeToCellHost.size() == numElements * numNbhPerElement);
    cudaMemcpy(target, nodeToCellHost.data(), sizeof(int) * numElements * numNbhPerElement,
               cudaMemcpyHostToDevice);
  }
}

GpuTriMesh::GpuTriMesh(const atlas::Mesh& mesh) {
  // position vector
  cudaMalloc((void**)&pos_, sizeof(double2) * mesh.nodes().size());
  // neighbor counts
  const int cellsPerNode = 6;
  const int edgesPerNode = 6;
  const int cellsPerEdge = 2;
  const int nodesPerEdge = 2;
  const int nodesPerCell = 3;
  const int edgesPerCell = 3;
  // nbh list allocation (could maybe be improved using mallocPitch)
  cudaMalloc((void**)&nodeToCell_, sizeof(int) * mesh.nodes().size() * cellsPerNode);
  cudaMalloc((void**)&nodeToEdge_, sizeof(int) * mesh.nodes().size() * edgesPerNode);
  cudaMalloc((void**)&edgeToCell_, sizeof(int) * mesh.edges().size() * cellsPerEdge);
  cudaMalloc((void**)&edgeToNode_, sizeof(int) * mesh.edges().size() * nodesPerEdge);
  cudaMalloc((void**)&cellToNode_, sizeof(int) * mesh.cells().size() * nodesPerCell);
  cudaMalloc((void**)&cellToEdge_, sizeof(int) * mesh.cells().size() * edgesPerCell);
  // copy position vector
  std::vector<double2> pHost;
  auto xy = atlas::array::make_view<double, 2>(mesh.nodes().xy());
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    pHost.push_back({xy(nodeIdx, atlas::LON), xy(nodeIdx, atlas::LAT)});
  }
  cudaMemcpy(pos_, pHost.data(), sizeof(double2) * mesh.nodes().size(), cudaMemcpyHostToDevice);
  // copy neighbor tables
  copyNeighborTable(mesh.nodes().cell_connectivity(), mesh.nodes().size(), cellsPerNode,
                    nodeToCell_);
  copyNeighborTable(mesh.nodes().edge_connectivity(), mesh.nodes().size(), edgesPerNode,
                    nodeToEdge_);
  copyNeighborTable(mesh.edges().cell_connectivity(), mesh.edges().size(), cellsPerEdge,
                    edgeToCell_);
  copyNeighborTable(mesh.edges().node_connectivity(), mesh.edges().size(), nodesPerEdge,
                    edgeToNode_);
  copyNeighborTable(mesh.cells().node_connectivity(), mesh.cells().size(), nodesPerCell,
                    cellToNode_);
  copyNeighborTable(mesh.cells().node_connectivity(), mesh.cells().size(), edgesPerCell,
                    cellToEdge_);
}

LaplacianStencil::LaplacianStencil(const atlas::Mesh& mesh,
                                   const atlasInterface::Field<double>& vec,
                                   const atlasInterface::Field<double>& rotVec,
                                   const atlasInterface::SparseDimension<double>& geofacRot,
                                   const atlasInterface::Field<double>& divVec,
                                   const atlasInterface::SparseDimension<double>& geofacDiv,
                                   const atlasInterface::Field<double>& primal_edge_length,
                                   const atlasInterface::Field<double>& dual_edge_length,
                                   const atlasInterface::Field<double>& tangent_orientation,
                                   const atlasInterface::Field<double>& nabla2vec)
    : mesh_(mesh) {

  printf("called succesfully!\n");

  // alloc fields
  cudaMalloc((void**)&vec_, sizeof(double) * vec.numElements());
  cudaMalloc((void**)&rotVec, sizeof(double) * rotVec.numElements());
  cudaMalloc((void**)&geofacRot_, sizeof(double) * geofacRot.numElements());
  cudaMalloc((void**)&divVec, sizeof(double) * divVec.numElements());
  cudaMalloc((void**)&geofacDiv_, sizeof(double) * geofacDiv.numElements());
  cudaMalloc((void**)&primal_edge_length_, sizeof(double) * primal_edge_length.numElements());
  cudaMalloc((void**)&dual_edge_length_, sizeof(double) * dual_edge_length.numElements());
  cudaMalloc((void**)&tangent_orientation_, sizeof(double) * tangent_orientation.numElements());
  cudaMalloc((void**)&nabla2vec_, sizeof(double) * nabla2vec.numElements());

  // copy fields from host to device
  cudaMemcpy(vec_, vec.data(), sizeof(double) * vec.numElements(), cudaMemcpyHostToDevice);
  cudaMemcpy(rotVec_, rotVec.data(), sizeof(double) * rotVec.numElements(), cudaMemcpyHostToDevice);
  cudaMemcpy(geofacRot_, geofacRot.data(), sizeof(double) * geofacRot.numElements(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(divVec_, divVec.data(), sizeof(double) * divVec.numElements(), cudaMemcpyHostToDevice);
  cudaMemcpy(geofacDiv_, geofacDiv.data(), sizeof(double) * geofacDiv.numElements(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(primal_edge_length_, primal_edge_length.data(),
             sizeof(double) * primal_edge_length.numElements(), cudaMemcpyHostToDevice);
  cudaMemcpy(dual_edge_length_, dual_edge_length.data(),
             sizeof(double) * dual_edge_length.numElements(), cudaMemcpyHostToDevice);
  cudaMemcpy(tangent_orientation_, tangent_orientation.data(),
             sizeof(double) * tangent_orientation.numElements(), cudaMemcpyHostToDevice);
  cudaMemcpy(nabla2vec_, nabla2vec.data(), sizeof(double) * nabla2vec.numElements(),
             cudaMemcpyHostToDevice);
}

void LaplacianStencil::run() {
  // stage over nodes
  {
    dim3 dG((mesh_.NumNodes() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);
    computeRot<<<dG, dB>>>(mesh_.EdgeToNode(), vec_, geofacRot_, mesh_.NumNodes(), rotVec_);
  }
  // stage over cells
  {
    dim3 dG((mesh_.NumCells() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);
    computeDiv<<<dG, dB>>>(mesh_.CellToEdge(), vec_, geofacDiv_, mesh_.NumCells(), divVec_);
  }
  // stage over edges
  {
    dim3 dG((mesh_.NumEdges() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);
    computeLapl<<<dG, dB>>>(mesh_.EdgeToNode(), rotVec_, primal_edge_length_, tangent_orientation_,
                            mesh_.EdgeToCell(), divVec_, dual_edge_length_, mesh_.NumEdges(),
                            nabla2vec_);
  }
}

void LaplacianStencil::CopyResultToHost(const atlasInterface::Field<double>& vec,
                                        atlasInterface::Field<double>& rotVec,
                                        atlasInterface::SparseDimension<double>& geofacRot,
                                        atlasInterface::Field<double>& divVec,
                                        atlasInterface::SparseDimension<double>& geofacDiv,
                                        atlasInterface::Field<double>& primal_edge_length,
                                        atlasInterface::Field<double>& dual_edge_length,
                                        atlasInterface::Field<double>& tangent_orientation,
                                        atlasInterface::Field<double>& nabla2vec) const {
  cudaMemcpy(vec.data(), vec_, sizeof(double) * vec.numElements(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rotVec.data(), rotVec_, sizeof(double) * rotVec.numElements(), cudaMemcpyDeviceToHost);
  cudaMemcpy(geofacRot.data(), geofacRot_, sizeof(double) * geofacRot.numElements(),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(divVec.data(), divVec_, sizeof(double) * divVec.numElements(), cudaMemcpyDeviceToHost);
  cudaMemcpy(geofacDiv.data(), geofacDiv_, sizeof(double) * geofacDiv.numElements(),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(primal_edge_length_, primal_edge_length.data(),
             sizeof(double) * primal_edge_length.numElements(), cudaMemcpyDeviceToHost);
  cudaMemcpy(dual_edge_length.data(), dual_edge_length_,
             sizeof(double) * dual_edge_length.numElements(), cudaMemcpyDeviceToHost);
  cudaMemcpy(tangent_orientation.data(), tangent_orientation_,
             sizeof(double) * tangent_orientation.numElements(), cudaMemcpyDeviceToHost);
  cudaMemcpy(nabla2vec.data(), nabla2vec_, sizeof(double) * nabla2vec.numElements(),
             cudaMemcpyDeviceToHost);
}
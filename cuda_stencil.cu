//===--------------------------------------------------------------------------------*-
// C++ -*-===//
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

#define gpuErrchk(ans)                                                                             \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if(code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort)
      exit(code);
  }
}

#define EDGES_PER_NODE 6
#define EDGES_PER_CELL 3
#define CELLS_PER_EDGE 2
#define NODES_PER_EDGE 2
#define BLOCK_SIZE 128
#define DEVICE_MISSING_VALUE -1
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
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
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
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
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
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
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
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs += __ldg(&divVec[nbhIdx]) * weights[nbhIter];
    }
    nabla2vec[pidx] += lhs;
  }
  nabla2vec[pidx] = nabla2vec[pidx] / __ldg(&dual_edge_length[pidx]);
}
} // namespace

template <typename ConnectivityT>
void copyNeighborTableIfPresent(const ConnectivityT& conn, int numElements, int numNbhPerElement,
                                int* target) {
  if(conn.rows() == 0) {
    return;
  }

  std::vector<int> hostTable;
  for(int elemIdx = 0; elemIdx < numElements; elemIdx++) {
    for(int nbhIdx = 0; nbhIdx < numNbhPerElement; nbhIdx++) {
      if(nbhIdx < conn.cols(elemIdx)) {
        hostTable.push_back(conn(elemIdx, nbhIdx));
      } else {
        hostTable.push_back(DEVICE_MISSING_VALUE);
      }
    }
  }
  assert(hostTable.size() == numElements * numNbhPerElement);
  gpuErrchk(cudaMemcpy(target, hostTable.data(), sizeof(int) * numElements * numNbhPerElement,
                       cudaMemcpyHostToDevice));
}

GpuTriMesh::GpuTriMesh(const atlas::Mesh& mesh) {
  // position vector
  gpuErrchk(cudaMalloc((void**)&pos_, sizeof(double2) * mesh.nodes().size()));
  // neighbor counts
  const int cellsPerNode = 6;
  const int edgesPerNode = 6;
  const int cellsPerEdge = 2;
  const int nodesPerEdge = 2;
  const int nodesPerCell = 3;
  const int edgesPerCell = 3;
  // nbh list allocation (could maybe be improved using mallocPitch)
  gpuErrchk(cudaMalloc((void**)&nodeToCell_, sizeof(int) * mesh.nodes().size() * cellsPerNode));
  gpuErrchk(cudaMalloc((void**)&nodeToEdge_, sizeof(int) * mesh.nodes().size() * edgesPerNode));
  gpuErrchk(cudaMalloc((void**)&edgeToCell_, sizeof(int) * mesh.edges().size() * cellsPerEdge));
  gpuErrchk(cudaMalloc((void**)&edgeToNode_, sizeof(int) * mesh.edges().size() * nodesPerEdge));
  gpuErrchk(cudaMalloc((void**)&cellToNode_, sizeof(int) * mesh.cells().size() * nodesPerCell));
  gpuErrchk(cudaMalloc((void**)&cellToEdge_, sizeof(int) * mesh.cells().size() * edgesPerCell));
  // copy position vector
  std::vector<double2> pHost;
  auto xy = atlas::array::make_view<double, 2>(mesh.nodes().xy());
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    pHost.push_back({xy(nodeIdx, atlas::LON), xy(nodeIdx, atlas::LAT)});
  }
  gpuErrchk(cudaMemcpy(pos_, pHost.data(), sizeof(double2) * mesh.nodes().size(),
                       cudaMemcpyHostToDevice));
  // copy neighbor tables
  copyNeighborTableIfPresent(mesh.nodes().cell_connectivity(), mesh.nodes().size(), cellsPerNode,
                             nodeToCell_);
  copyNeighborTableIfPresent(mesh.nodes().edge_connectivity(), mesh.nodes().size(), edgesPerNode,
                             nodeToEdge_);
  copyNeighborTableIfPresent(mesh.edges().cell_connectivity(), mesh.edges().size(), cellsPerEdge,
                             edgeToCell_);
  copyNeighborTableIfPresent(mesh.edges().node_connectivity(), mesh.edges().size(), nodesPerEdge,
                             edgeToNode_);
  copyNeighborTableIfPresent(mesh.cells().node_connectivity(), mesh.cells().size(), nodesPerCell,
                             cellToNode_);
  copyNeighborTableIfPresent(mesh.cells().node_connectivity(), mesh.cells().size(), edgesPerCell,
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
  gpuErrchk(cudaMalloc((void**)&vec_, sizeof(double) * vec.numElements()));
  gpuErrchk(cudaMalloc((void**)&rotVec, sizeof(double) * rotVec.numElements()));
  gpuErrchk(cudaMalloc((void**)&geofacRot_, sizeof(double) * geofacRot.numElements()));
  gpuErrchk(cudaMalloc((void**)&divVec, sizeof(double) * divVec.numElements()));
  gpuErrchk(cudaMalloc((void**)&geofacDiv_, sizeof(double) * geofacDiv.numElements()));
  gpuErrchk(
      cudaMalloc((void**)&primal_edge_length_, sizeof(double) * primal_edge_length.numElements()));
  gpuErrchk(
      cudaMalloc((void**)&dual_edge_length_, sizeof(double) * dual_edge_length.numElements()));
  gpuErrchk(cudaMalloc((void**)&tangent_orientation_,
                       sizeof(double) * tangent_orientation.numElements()));
  gpuErrchk(cudaMalloc((void**)&nabla2vec_, sizeof(double) * nabla2vec.numElements()));

  // copy fields from host to device
  gpuErrchk(
      cudaMemcpy(vec_, vec.data(), sizeof(double) * vec.numElements(), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(rotVec_, rotVec.data(), sizeof(double) * rotVec.numElements(),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(geofacRot_, geofacRot.data(), sizeof(double) * geofacRot.numElements(),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(divVec_, divVec.data(), sizeof(double) * divVec.numElements(),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(geofacDiv_, geofacDiv.data(), sizeof(double) * geofacDiv.numElements(),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(primal_edge_length_, primal_edge_length.data(),
                       sizeof(double) * primal_edge_length.numElements(), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dual_edge_length_, dual_edge_length.data(),
                       sizeof(double) * dual_edge_length.numElements(), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(tangent_orientation_, tangent_orientation.data(),
                       sizeof(double) * tangent_orientation.numElements(), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(nabla2vec_, nabla2vec.data(), sizeof(double) * nabla2vec.numElements(),
                       cudaMemcpyHostToDevice));
}

void LaplacianStencil::run() {
  // stage over nodes
  {
    dim3 dG((mesh_.NumNodes() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);
    computeRot<<<dG, dB>>>(mesh_.EdgeToNode(), vec_, geofacRot_, mesh_.NumNodes(), rotVec_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
  // stage over cells
  {
    dim3 dG((mesh_.NumCells() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);
    computeDiv<<<dG, dB>>>(mesh_.CellToEdge(), vec_, geofacDiv_, mesh_.NumCells(), divVec_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
  // stage over edges
  {
    dim3 dG((mesh_.NumEdges() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);
    computeLapl<<<dG, dB>>>(mesh_.EdgeToNode(), rotVec_, primal_edge_length_, tangent_orientation_,
                            mesh_.EdgeToCell(), divVec_, dual_edge_length_, mesh_.NumEdges(),
                            nabla2vec_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
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
  gpuErrchk(cudaMemcpy((double*)vec.data(), vec_, sizeof(double) * vec.numElements(),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)rotVec.data(), rotVec_, sizeof(double) * rotVec.numElements(),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)geofacRot.data(), geofacRot_,
                       sizeof(double) * geofacRot.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)divVec.data(), divVec_, sizeof(double) * divVec.numElements(),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)geofacDiv.data(), geofacDiv_,
                       sizeof(double) * geofacDiv.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)primal_edge_length_, primal_edge_length.data(),
                       sizeof(double) * primal_edge_length.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)dual_edge_length.data(), dual_edge_length_,
                       sizeof(double) * dual_edge_length.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)tangent_orientation.data(), tangent_orientation_,
                       sizeof(double) * tangent_orientation.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)nabla2vec.data(), nabla2vec_,
                       sizeof(double) * nabla2vec.numElements(), cudaMemcpyDeviceToHost));
}
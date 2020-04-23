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

#include "diamond_stencil.h"

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
#define E_C_V_SIZE 4
#define BLOCK_SIZE 128
#define DEVICE_MISSING_VALUE -1
// __global__ void computeRot(const int* __restrict__ nodeToEdge, const double* __restrict__ vecE,
//                            const double* __restrict__ geofacRot, int numNodes, double* rotVec) {
//   unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
//   if(pidx >= numNodes) {
//     return;
//   }
//   {
//     double lhs = 0.;                                            // init
//     for(int nbhIter = 0; nbhIter < EDGES_PER_NODE; nbhIter++) { // reduceEdgeToVertex
//       int nbhIdx = __ldg(&nodeToEdge[pidx * EDGES_PER_NODE + nbhIter]);
//       if(nbhIdx == DEVICE_MISSING_VALUE) {
//         continue;
//       }
//       lhs += __ldg(&vecE[nbhIdx]) * __ldg(&geofacRot[pidx * EDGES_PER_NODE + nbhIter]);
//     }
//     rotVec[pidx] = lhs;
//   }
// }
// __global__ void computeDiv(const int* __restrict__ cellToEdge, const double* __restrict__ vecE,
//                            const double* __restrict__ geofacDiv, int numCells, double* divVec) {
//   unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
//   if(pidx >= numCells) {
//     return;
//   }
//   {
//     double lhs = 0.;                                            // init
//     for(int nbhIter = 0; nbhIter < EDGES_PER_CELL; nbhIter++) { // reduceEdgeToCell
//       int nbhIdx = __ldg(&cellToEdge[pidx * EDGES_PER_CELL + nbhIter]);
//       if(nbhIdx == DEVICE_MISSING_VALUE) {
//         continue;
//       }
//       lhs += __ldg(&vecE[nbhIdx]) * __ldg(&geofacDiv[pidx * EDGES_PER_CELL + nbhIter]);
//     }
//     divVec[pidx] = lhs;
//   }
// }
__global__ void computeVn(int numEdges, const int* __restrict__ ecvTable,
                          double* __restrict__ vn_vert, const double* __restrict__ u_vert,
                          const double* __restrict__ v_vert,
                          const double* __restrict__ primal_normal_vert_x,
                          const double* __restrict__ primal_normal_vert_y) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  {
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      vn_vert[pidx * E_C_V_SIZE + nbhIter] =
          __ldg(&u_vert[nbhIdx]) * __ldg(&primal_normal_vert_x[pidx * E_C_V_SIZE + nbhIter]) +
          __ldg(&v_vert[nbhIdx]) * __ldg(&primal_normal_vert_y[pidx * E_C_V_SIZE + nbhIter]);
    }
  }
}
} // namespace

void generateNbhTable(atlas::Mesh const& mesh, std::vector<dawn::LocationType> chain,
                      int numElements, int numNbhPerElement, int* target) {
  std::vector<int> elems;
  switch(chain.front()) {
  case dawn::LocationType::Cells:
    elems = getCells(atlasTag(), mesh);
    break;
  case dawn::LocationType::Edges:
    elems = getEdges(atlasTag(), mesh);
    break;
  case dawn::LocationType::Vertices:
    elems = getVertices(atlasTag(), mesh);
    break;
  }

  assert(elems.size() == numElements);

  std::vector<int> hostTable;
  for(int elem : elems) {
    auto neighbors = getNeighbors(mesh, chain, elem);
    for(int nbhIdx = 0; nbhIdx < numNbhPerElement; nbhIdx++) {
      if(nbhIdx < neighbors.size()) {
        hostTable.push_back(neighbors[nbhIdx]);
      } else {
        hostTable.push_back(DEVICE_MISSING_VALUE);
      }
    }
  }

  assert(hostTable.size() == numElements * numNbhPerElement);
  gpuErrchk(cudaMemcpy(target, hostTable.data(), sizeof(int) * numElements * numNbhPerElement,
                       cudaMemcpyHostToDevice));
}

// template <typename ConnectivityT>
// void copyNeighborTableIfPresent(const ConnectivityT& conn, int numElements, int numNbhPerElement,
//                                 int* target) {
//   if(conn.rows() == 0) {
//     return;
//   }

//   std::vector<int> hostTable;
//   for(int elemIdx = 0; elemIdx < numElements; elemIdx++) {
//     for(int nbhIdx = 0; nbhIdx < numNbhPerElement; nbhIdx++) {
//       if(nbhIdx < conn.cols(elemIdx)) {
//         hostTable.push_back(conn(elemIdx, nbhIdx));
//       } else {
//         hostTable.push_back(DEVICE_MISSING_VALUE);
//       }
//     }
//   }
//   assert(hostTable.size() == numElements * numNbhPerElement);
//   gpuErrchk(cudaMemcpy(target, hostTable.data(), sizeof(int) * numElements * numNbhPerElement,
//                        cudaMemcpyHostToDevice));
// }

GpuTriMesh::GpuTriMesh(const atlas::Mesh& mesh) {

  numNodes_ = mesh.nodes().size();
  numEdges_ = mesh.edges().size();
  numCells_ = mesh.cells().size();

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
  // gpuErrchk(cudaMalloc((void **)&nodeToCell_, sizeof(int) * mesh.nodes().size() * cellsPerNode));
  // gpuErrchk(cudaMalloc((void **)&nodeToEdge_, sizeof(int) * mesh.nodes().size() * edgesPerNode));
  // gpuErrchk(cudaMalloc((void **)&edgeToCell_, sizeof(int) * mesh.edges().size() * cellsPerEdge));
  // gpuErrchk(cudaMalloc((void **)&edgeToNode_, sizeof(int) * mesh.edges().size() * nodesPerEdge));
  // gpuErrchk(cudaMalloc((void **)&cellToNode_, sizeof(int) * mesh.cells().size() * nodesPerCell));
  // gpuErrchk(cudaMalloc((void **)&cellToEdge_, sizeof(int) * mesh.cells().size() * edgesPerCell));
  gpuErrchk(cudaMalloc((void**)&ecvTable_, sizeof(int) * mesh.edges().size() * E_C_V_SIZE));
  // copy position vector
  std::vector<double2> pHost;
  auto xy = atlas::array::make_view<double, 2>(mesh.nodes().xy());
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    pHost.push_back({xy(nodeIdx, atlas::LON), xy(nodeIdx, atlas::LAT)});
  }
  gpuErrchk(cudaMemcpy(pos_, pHost.data(), sizeof(double2) * mesh.nodes().size(),
                       cudaMemcpyHostToDevice));
  generateNbhTable(
      mesh, {dawn::LocationType::Edges, dawn::LocationType::Cells, dawn::LocationType::Vertices},
      mesh.edges().size(), E_C_V_SIZE, ecvTable_);
  // copy neighbor tables
  // copyNeighborTableIfPresent(mesh.nodes().cell_connectivity(), mesh.nodes().size(), cellsPerNode,
  //                            nodeToCell_);
  // copyNeighborTableIfPresent(mesh.nodes().edge_connectivity(), mesh.nodes().size(), edgesPerNode,
  //                            nodeToEdge_);
  // copyNeighborTableIfPresent(mesh.edges().cell_connectivity(), mesh.edges().size(), cellsPerEdge,
  //                            edgeToCell_);
  // copyNeighborTableIfPresent(mesh.edges().node_connectivity(), mesh.edges().size(), nodesPerEdge,
  //                            edgeToNode_);
  // copyNeighborTableIfPresent(mesh.cells().node_connectivity(), mesh.cells().size(), nodesPerCell,
  //                            cellToNode_);
  // copyNeighborTableIfPresent(mesh.cells().edge_connectivity(), mesh.cells().size(), edgesPerCell,
  //                            cellToEdge_);
}

#define initField(field, cudaStorage)                                                              \
  {                                                                                                \
    gpuErrchk(cudaMalloc((void**)&cudaStorage, sizeof(double) * field.numElements()));             \
    gpuErrchk(cudaMemcpy(cudaStorage, field.data(), sizeof(double) * field.numElements(),          \
                         cudaMemcpyHostToDevice));                                                 \
  }

DiamondStencil::diamond_stencil::diamond_stencil(
    const atlas::Mesh& mesh, const atlasInterface::Field<double>& diff_multfac_smag,
    const atlasInterface::Field<double>& u_vert, const atlasInterface::Field<double>& v_vert,
    const atlasInterface::Field<double>& tangent_orientation,
    const atlasInterface::Field<double>& inv_primal_edge_length,
    const atlasInterface::Field<double>& inv_vert_vert_length,
    const atlasInterface::Field<double>& dvt_tang, const atlasInterface::Field<double>& dvt_norm,
    const atlasInterface::SparseDimension<double>& vn_vert,
    const atlasInterface::SparseDimension<double>& primal_normal_vert_x,
    const atlasInterface::SparseDimension<double>& primal_normal_vert_y,
    const atlasInterface::SparseDimension<double>& dual_normal_vert_x,
    const atlasInterface::SparseDimension<double>& dual_normal_vert_y,
    const atlasInterface::Field<double>& kh_smag_e, const atlasInterface::Field<double>& kh_smag_ec,
    const atlasInterface::Field<double>& z_nabla2_e)
    : sbase("diamond_stencil"), mesh_(mesh) {

  initField(diff_multfac_smag, diff_multfac_smag_);
  initField(u_vert, u_vert_);
  initField(v_vert, v_vert_);
  initField(tangent_orientation, tangent_orientation_);
  initField(inv_primal_edge_length, inv_primal_edge_length_);
  initField(inv_vert_vert_length, inv_vert_vert_length_);
  initField(dvt_tang, dvt_tang_);
  initField(dvt_norm, dvt_norm_);
  initField(vn_vert, vn_vert_);
  initField(primal_normal_vert_x, primal_normal_vert_x_);
  initField(primal_normal_vert_y, primal_normal_vert_y_);
  initField(dual_normal_vert_x, dual_normal_vert_x_);
  initField(dual_normal_vert_y, dual_normal_vert_y_);
  initField(kh_smag_e, kh_smag_e_);
  initField(kh_smag_ec, kh_smag_ec_);
  initField(z_nabla2_e, z_nabla2_e_);
}

void DiamondStencil::diamond_stencil::run() {
  // starting timers
  start();

  // stage over edges
  {
    dim3 dG((mesh_.NumEdges() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);
    computeVn<<<dG, dB>>>(mesh_.NumEdges(), ecvTable_, vn_vert_, u_vert_, v_vert_,
                          primal_normal_vert_x_, primal_normal_vert_y_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  // stopping timers
  pause();
}

void DiamondStencil::diamond_stencil::CopyResultToHost(
    atlasInterface::Field<double>& kh_smag_e, atlasInterface::Field<double>& kh_smag_ec,
    atlasInterface::Field<double>& z_nabla2_e) const {
  gpuErrchk(cudaMemcpy((double*)kh_smag_e.data(), kh_smag_e_,
                       sizeof(double) * kh_smag_e.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)kh_smag_ec.data(), kh_smag_ec_,
                       sizeof(double) * kh_smag_ec.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)z_nabla2_e.data(), z_nabla2_e_,
                       sizeof(double) * z_nabla2_e.numElements(), cudaMemcpyDeviceToHost));
}

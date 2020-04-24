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

#include <vector>

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

#define E_C_V_SIZE 4
#define BLOCK_SIZE 128
#define DEVICE_MISSING_VALUE -1

__global__ void compute_vn(int numEdges, int kSize, const int* __restrict__ ecvTable,
                           double* __restrict__ vn_vert, const double* __restrict__ u_vert,
                           const double* __restrict__ v_vert,
                           const double* __restrict__ primal_normal_vert_x,
                           const double* __restrict__ primal_normal_vert_y) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  {
    for(int kIter = 0; kIter < kSize; kIter++) {
      int offset = kIter * numEdges;
      for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
        int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
        if(nbhIdx == DEVICE_MISSING_VALUE) {
          continue;
        }
        vn_vert[offset + pidx * E_C_V_SIZE + nbhIter] =
            __ldg(&u_vert[offset + nbhIdx]) *
                __ldg(&primal_normal_vert_x[offset + pidx * E_C_V_SIZE + nbhIter]) +
            __ldg(&v_vert[offset + nbhIdx]) *
                __ldg(&primal_normal_vert_y[offset + pidx * E_C_V_SIZE + nbhIter]);
      }
    }
  }
}

__global__ void reduce_dvt_tang(int numEdges, const int* __restrict__ ecvTable,
                                double* __restrict__ dvt_tang, const double* __restrict__ u_vert,
                                const double* __restrict__ v_vert,
                                const double* __restrict__ dual_normal_vert_x,
                                const double* __restrict__ dual_normal_vert_y) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  double weights[E_C_V_SIZE] = {-1., 1., 0., 0.};
  {
    double lhs = 0.;
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs += weights[nbhIter] * __ldg(&u_vert[nbhIdx]) *
                 __ldg(&dual_normal_vert_x[pidx * E_C_V_SIZE + nbhIter]) +
             __ldg(&v_vert[nbhIdx]) * __ldg(&dual_normal_vert_y[pidx * E_C_V_SIZE + nbhIter]);
    }
    dvt_tang[pidx] = lhs;
  }
}

__global__ void finish_dvt_tang(int numEdges, double* __restrict__ dvt_tang,
                                const double* __restrict__ tangent_orientation) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  dvt_tang[pidx] = dvt_tang[pidx] * __ldg(&tangent_orientation[pidx]);
}

__global__ void reduce_dvt_norm(int numEdges, const int* __restrict__ ecvTable,
                                double* __restrict__ dvt_norm, const double* __restrict__ u_vert,
                                const double* __restrict__ v_vert,
                                const double* __restrict__ dual_normal_vert_x,
                                const double* __restrict__ dual_normal_vert_y) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  double weights[E_C_V_SIZE] = {0., 0., -1., 1.};
  {
    double lhs = 0.;
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs += weights[nbhIter] * __ldg(&u_vert[nbhIdx]) *
                 __ldg(&dual_normal_vert_x[pidx * E_C_V_SIZE + nbhIter]) +
             __ldg(&v_vert[nbhIdx]) * __ldg(&dual_normal_vert_y[pidx * E_C_V_SIZE + nbhIter]);
    }
    dvt_norm[pidx] = lhs;
  }
}

__global__ void smagorinsky_1(int numEdges, const int* __restrict__ ecvTable,
                              double* __restrict__ kh_smag_1, const double* __restrict__ vn_vert) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  double weights[E_C_V_SIZE] = {-1., 1., 0., 0.};
  {
    double lhs = 0.;
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs += vn_vert[nbhIdx] * weights[nbhIter];
    }
    kh_smag_1[pidx] = lhs;
  }
} // namespace

__global__ void smagorinsky_1_multitply_facs(int numEdges, double* __restrict__ kh_smag_1,
                                             const double* __restrict__ tangent_orientation,
                                             const double* __restrict__ inv_vert_vert_length,
                                             const double* __restrict__ inv_primal_edge_length,
                                             const double* __restrict__ dvt_norm) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  kh_smag_1[pidx] =
      kh_smag_1[pidx] * __ldg(&inv_primal_edge_length[pidx]) * __ldg(&tangent_orientation[pidx]) -
      __ldg(&dvt_norm[pidx]) * __ldg(&inv_vert_vert_length[pidx]);
}

__global__ void smagorinsky_1_square(int numEdges, double* __restrict__ kh_smag_1) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  kh_smag_1[pidx] = kh_smag_1[pidx] * kh_smag_1[pidx];
}

__global__ void smagorinsky_2(int numEdges, const int* __restrict__ ecvTable,
                              double* __restrict__ kh_smag_2, const double* __restrict__ vn_vert) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  double weights[E_C_V_SIZE] = {0., 0., -1., 1.};
  {
    double lhs = 0.;
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs += vn_vert[nbhIdx] * weights[nbhIter];
    }
    kh_smag_2[pidx] = lhs;
  }
}

__global__ void smagorinsky_2_multitply_facs(int numEdges, double* __restrict__ kh_smag_2,
                                             const double* __restrict__ inv_vert_vert_length,
                                             const double* __restrict__ inv_primal_edge_length,
                                             const double* __restrict__ dvt_tang) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  kh_smag_2[pidx] = kh_smag_2[pidx] * __ldg(&inv_vert_vert_length[pidx]) -
                    __ldg(&dvt_tang[pidx]) * __ldg(&inv_primal_edge_length[pidx]);
}

__global__ void smagorinsky_2_square(int numEdges, double* __restrict__ kh_smag_2) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  kh_smag_2[pidx] = kh_smag_2[pidx] * kh_smag_2[pidx];
}

__global__ void smagorinsky(int numEdges, double* __restrict__ kh_smag,
                            const double* __restrict__ kh_smag_1,
                            const double* __restrict__ kh_smag_2) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  kh_smag[pidx] = sqrt(kh_smag_1[pidx] + kh_smag_2[pidx]);
}

__global__ void diamond(int numEdges, int kSize, const int* __restrict__ ecvTable,
                        double* __restrict__ nabla2, const double* __restrict__ vn_vert,
                        const double* __restrict__ inv_primal_edge_length,
                        const double* __restrict__ inv_vert_vert_length) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  double weights[E_C_V_SIZE] = {
      __ldg(&inv_primal_edge_length[pidx]) * __ldg(&inv_primal_edge_length[pidx]),
      __ldg(&inv_primal_edge_length[pidx]) * __ldg(&inv_primal_edge_length[pidx]),
      __ldg(&inv_vert_vert_length[pidx]) * __ldg(&inv_vert_vert_length[pidx]),
      __ldg(&inv_vert_vert_length[pidx]) * __ldg(&inv_vert_vert_length[pidx])};

  for(int kIter = 0; kIter < kSize; kIter++) {
    int offset = kIter * numEdges;
    double lhs = 0.;
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
      int nbhIdx = __ldg(&ecvTable[offset + pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs += 4. * vn_vert[offset + pidx * E_C_V_SIZE + nbhIter] * weights[offset + nbhIter];
    }
    nabla2[pidx] = lhs;
  }
}

__global__ void nabla2(int numEdges, int kSize, double* __restrict__ nabla2,
                       double* __restrict__ vn, const double* __restrict__ inv_primal_edge_length,
                       const double* __restrict__ inv_vert_vert_length) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pidx >= numEdges) {
    return;
  }
  for(int kIter = 0; kIter < kSize; kIter++) {
    int offset = kIter * numEdges;
    nabla2[offset + pidx] =
        nabla2[offset + pidx] -
        8. * __ldg(&vn[offset + pidx]) * __ldg(&inv_primal_edge_length[offset + pidx]) *
            __ldg(&inv_primal_edge_length[offset + pidx]) -
        8. * __ldg(&vn[offset + pidx]) * __ldg(&inv_vert_vert_length[offset + pidx]) *
            __ldg(&inv_vert_vert_length[offset + pidx]);
  }
}

} // namespace

void generateNbhTable(atlas::Mesh const& mesh, std::vector<dawn::LocationType> chain,
                      int numElements, int numNbhPerElement, int* target) {
  std::vector<atlas::idx_t> elems;
  switch(chain.front()) {
  case dawn::LocationType::Cells: {
    for(auto cell : atlasInterface::getCells(atlasInterface::atlasTag(), mesh)) {
      elems.push_back(cell);
    }
    break;
  }
  case dawn::LocationType::Edges: {
    for(auto edge : atlasInterface::getEdges(atlasInterface::atlasTag(), mesh)) {
      elems.push_back(edge);
    }
    break;
  }
  case dawn::LocationType::Vertices: {
    for(auto vertex : atlasInterface::getVertices(atlasInterface::atlasTag(), mesh)) {
      elems.push_back(vertex);
    }
    break;
  }
  }

  assert(elems.size() == numElements);

  std::vector<int> hostTable;
  for(int elem : elems) {
    auto neighbors = atlasInterface::getNeighbors(mesh, chain, elem);
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

GpuTriMesh::GpuTriMesh(const atlas::Mesh& mesh) {

  numNodes_ = mesh.nodes().size();
  numEdges_ = mesh.edges().size();
  numCells_ = mesh.cells().size();

  // position vector
  gpuErrchk(cudaMalloc((void**)&pos_, sizeof(double2) * mesh.nodes().size()));

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
}

#define initField(field, cudaStorage)                                                              \
  {                                                                                                \
    gpuErrchk(cudaMalloc((void**)&cudaStorage, sizeof(double) * field.numElements()));             \
    gpuErrchk(cudaMemcpy(cudaStorage, field.data(), sizeof(double) * field.numElements(),          \
                         cudaMemcpyHostToDevice));                                                 \
  }

DiamondStencil::diamond_stencil::diamond_stencil(
    const atlas::Mesh& mesh, int k_size, const atlasInterface::Field<double>& diff_multfac_smag,
    const atlasInterface::Field<double>& tangent_orientation,
    const atlasInterface::Field<double>& inv_primal_edge_length,
    const atlasInterface::Field<double>& inv_vert_vert_length,
    const atlasInterface::Field<double>& u_vert, const atlasInterface::Field<double>& v_vert,
    const atlasInterface::SparseDimension<double>& primal_normal_vert_x,
    const atlasInterface::SparseDimension<double>& primal_normal_vert_y,
    const atlasInterface::SparseDimension<double>& dual_normal_vert_x,
    const atlasInterface::SparseDimension<double>& dual_normal_vert_y,
    const atlasInterface::SparseDimension<double>& vn_vert, const atlasInterface::Field<double>& vn,
    const atlasInterface::Field<double>& dvt_tang, const atlasInterface::Field<double>& dvt_norm,
    const atlasInterface::Field<double>& kh_smag_1, const atlasInterface::Field<double>& kh_smag_2,
    const atlasInterface::Field<double>& kh_smag_e, const atlasInterface::Field<double>& z_nabla2_e)
    : sbase("diamond_stencil"), mesh_(mesh), kSize_(k_size) {

  initField(diff_multfac_smag, diff_multfac_smag_);
  initField(tangent_orientation, tangent_orientation_);
  initField(inv_primal_edge_length, inv_primal_edge_length_);
  initField(inv_vert_vert_length, inv_vert_vert_length_);
  initField(u_vert, u_vert_);
  initField(v_vert, v_vert_);
  initField(primal_normal_vert_x, primal_normal_vert_x_);
  initField(primal_normal_vert_y, primal_normal_vert_y_);
  initField(dual_normal_vert_x, dual_normal_vert_x_);
  initField(dual_normal_vert_y, dual_normal_vert_y_);
  initField(vn_vert, vn_vert_);
  initField(vn, vn_);
  initField(dvt_tang, dvt_tang_);
  initField(dvt_norm, dvt_norm_);
  initField(kh_smag_1, kh_smag_1_);
  initField(kh_smag_2, kh_smag_2_);
  initField(kh_smag_e, kh_smag_e_);
  initField(z_nabla2_e, z_nabla2_e_);
}

void DiamondStencil::diamond_stencil::run() {
  // starting timers
  start();

  // stage over edges
  {
    dim3 dG((mesh_.NumEdges() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dB(BLOCK_SIZE);

    compute_vn<<<dG, dB>>>(mesh_.NumEdges(), kSize_, mesh_.ECVTable(), vn_vert_, u_vert_, v_vert_,
                           primal_normal_vert_x_, primal_normal_vert_y_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    reduce_dvt_tang<<<dG, dB>>>(mesh_.NumEdges(), mesh_.ECVTable(), dvt_tang_, u_vert_, v_vert_,
                                dual_normal_vert_x_, dual_normal_vert_y_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    finish_dvt_tang<<<dG, dB>>>(mesh_.NumEdges(), dvt_tang_, tangent_orientation_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    reduce_dvt_norm<<<dG, dB>>>(mesh_.NumEdges(), mesh_.ECVTable(), dvt_norm_, u_vert_, v_vert_,
                                dual_normal_vert_x_, dual_normal_vert_y_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    smagorinsky_1<<<dG, dB>>>(mesh_.NumEdges(), mesh_.ECVTable(), kh_smag_1_, vn_vert_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    smagorinsky_1_multitply_facs<<<dG, dB>>>(mesh_.NumEdges(), kh_smag_1_, tangent_orientation_,
                                             inv_vert_vert_length_, inv_primal_edge_length_,
                                             dvt_norm_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    smagorinsky_1_square<<<dG, dB>>>(mesh_.NumEdges(), kh_smag_1_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    smagorinsky_2<<<dG, dB>>>(mesh_.NumEdges(), mesh_.ECVTable(), kh_smag_2_, vn_vert_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    smagorinsky_2_multitply_facs<<<dG, dB>>>(mesh_.NumEdges(), kh_smag_2_, inv_vert_vert_length_,
                                             inv_primal_edge_length_, dvt_norm_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    smagorinsky_2_square<<<dG, dB>>>(mesh_.NumEdges(), kh_smag_2_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    smagorinsky<<<dG, dB>>>(mesh_.NumEdges(), kh_smag_e_, kh_smag_1_, kh_smag_2_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    diamond<<<dG, dB>>>(mesh_.NumEdges(), kSize_, mesh_.ECVTable(), z_nabla2_e_, vn_vert_,
                        inv_primal_edge_length_, inv_vert_vert_length_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    nabla2<<<dG, dB>>>(mesh_.NumEdges(), kSize_, z_nabla2_e_, vn_, inv_primal_edge_length_,
                       inv_vert_vert_length_);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  // stopping timers
  pause();
}

void DiamondStencil::diamond_stencil::CopyResultToHost(
    atlasInterface::Field<double>& kh_smag_e, atlasInterface::Field<double>& z_nabla2_e) const {
  gpuErrchk(cudaMemcpy((double*)kh_smag_e.data(), kh_smag_e_,
                       sizeof(double) * kh_smag_e.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((double*)z_nabla2_e.data(), z_nabla2_e_,
                       sizeof(double) * z_nabla2_e.numElements(), cudaMemcpyDeviceToHost));
}

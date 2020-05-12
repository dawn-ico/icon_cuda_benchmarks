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
#include "driver-includes/defs.hpp"

#include <atlas/mesh/Elements.h>
#include <atlas/mesh/HybridElements.h>
#include <atlas/mesh/Nodes.h>
#include <atlas/util/CoordinateEnums.h>

#include <assert.h>
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
#define BLOCK_SIZE 32
#define DEVICE_MISSING_VALUE -1
#define LEVELS_PER_THREAD 1

__global__ void merged(int numEdges, int numVertices, int kSize, const int* __restrict__ ecvTable,
                       dawn::float_type* __restrict__ nabla2,
                       const dawn::float_type* __restrict__ inv_primal_edge_length,
                       const dawn::float_type* __restrict__ inv_vert_vert_length,
                       const dawn::float_type* tangent_orientation, const float2* __restrict__ uv,
                       const float2* __restrict__ primal_normal_vert,
                       const float2* __restrict__ dual_normal_vert,
                       const dawn::float_type* __restrict__ vn,
                       const dawn::float_type* __restrict__ smag_fac,
                       dawn::float_type* __restrict__ kh_smag) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD;
  int khi = (kidx + 1) * LEVELS_PER_THREAD;
  if(pidx >= numEdges) {
    return;
  }
  const dawn::float_type weights_1[E_C_V_SIZE] = {-1., 1., 0., 0.};
  const dawn::float_type weights_2[E_C_V_SIZE] = {0., 0., -1., 1.};
  const dawn::float_type weights_tang[E_C_V_SIZE] = {-1., 1., 0., 0.};
  const dawn::float_type weights_norm[E_C_V_SIZE] = {0., 0., -1., 1.};
  {
    for(int kIter = klo; kIter < khi; kIter++) {
      if(kIter >= kSize) {
        return;
      }
      const int edgesDenseKOffset = kIter * numEdges;
      const int verticesDenseKOffset = kIter * numVertices;
      const int ecvSparseKOffset = kIter * numEdges * E_C_V_SIZE;
      const int denseIdx = edgesDenseKOffset + pidx;

      const dawn::float_type __local_inv_primal = inv_primal_edge_length[denseIdx];
      const dawn::float_type __local_inv_vert = inv_vert_vert_length[denseIdx];

      const dawn::float_type weights_nabla[E_C_V_SIZE] = {
          __local_inv_primal * __local_inv_primal, __local_inv_primal * __local_inv_primal,
          __local_inv_vert * __local_inv_vert, __local_inv_vert * __local_inv_vert};

      dawn::float_type lhs_1 = 0.;
      dawn::float_type lhs_2 = 0.;
      dawn::float_type lhs_nabla = 0.;
      dawn::float_type lhs_tang = 0.;
      dawn::float_type lhs_norm = 0.;
      for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) { // for(e->c->v)
        int nbhIdx = ecvTable[pidx * E_C_V_SIZE + nbhIter];
        if(nbhIdx == DEVICE_MISSING_VALUE) {
          continue;
        }
        const int sparseIdx = ecvSparseKOffset + nbhIter * numEdges + pidx;
        float2 __local_uv = uv[verticesDenseKOffset + nbhIdx];
        float2 __local_primal_normal_vert = primal_normal_vert[sparseIdx];
        dawn::float_type __local_vn_vert = __local_uv.x * __local_primal_normal_vert.x +
                                           __local_uv.y * __local_primal_normal_vert.y;
        lhs_1 += __local_vn_vert * weights_1[nbhIter];
        lhs_2 += __local_vn_vert * weights_2[nbhIter];
        lhs_nabla += 4. * __local_vn_vert * weights_nabla[nbhIter];
        float2 __local_dual_normal_vert = dual_normal_vert[sparseIdx];
        dawn::float_type tang_norm_rhs =
            (__local_uv.x * __local_dual_normal_vert.x + __local_uv.y * __local_dual_normal_vert.y);
        lhs_tang += weights_tang[nbhIter] * tang_norm_rhs;
        lhs_norm += weights_norm[nbhIter] * tang_norm_rhs;
      }
      const dawn::float_type __local_tangent_orientation = tangent_orientation[denseIdx];
      lhs_tang = lhs_tang * __local_tangent_orientation;

      const dawn::float_type rhs_smag_1 =
          lhs_1 * __local_inv_primal * __local_tangent_orientation + lhs_norm * __local_inv_vert;
      const dawn::float_type kh_smag_1 = rhs_smag_1 * rhs_smag_1;

      const dawn::float_type rhs_smag_2 = lhs_2 * __local_inv_vert + lhs_tang * __local_inv_primal;
      const dawn::float_type kh_smag_2 = rhs_smag_2 * rhs_smag_2;

      const dawn::float_type __local_vn = vn[denseIdx];

      nabla2[denseIdx] = lhs_nabla - 8. * __local_vn * __local_inv_primal * __local_inv_primal -
                         8. * __local_vn * __local_inv_vert * __local_inv_vert;

      kh_smag[denseIdx] = smag_fac[denseIdx] * sqrt(kh_smag_1 + kh_smag_2);
    }
  }
} // namespace

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

  gpuErrchk(cudaMalloc((void**)&ecvTable_, sizeof(int) * mesh.edges().size() * E_C_V_SIZE));

  generateNbhTable(
      mesh, {dawn::LocationType::Edges, dawn::LocationType::Cells, dawn::LocationType::Vertices},
      mesh.edges().size(), E_C_V_SIZE, ecvTable_);
}

void reshape(const dawn::float_type* input, dawn::float_type* output, int kSize, int numEdges,
             int sparseSize) {
  // In: edges, klevels, sparse
  // Out: klevels, sparse, edges

  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++)
      for(int sparseIdx = 0; sparseIdx < sparseSize; sparseIdx++) {
        output[kLevel * numEdges * sparseSize + sparseIdx * numEdges + edgeIdx] =
            input[edgeIdx * kSize * sparseSize + kLevel * sparseSize + sparseIdx];
      }
}

void reshape(const dawn::float_type* input, dawn::float_type* output, int kSize, int numEdges) {
  // In: edges, klevels
  // Out: klevels, edges

  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++) {
      output[kLevel * numEdges + edgeIdx] = input[edgeIdx * kSize + kLevel];
    }
}

void reshape_back(const dawn::float_type* input, dawn::float_type* output, int kSize,
                  int numEdges) {
  // In: klevels, edges
  // Out: edges, klevels

  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
    for(int kLevel = 0; kLevel < kSize; kLevel++) {
      output[edgeIdx * kSize + kLevel] = input[kLevel * numEdges + edgeIdx];
    }
}
void initField(const atlasInterface::Field<dawn::float_type>& field, dawn::float_type** cudaStorage,
               int denseSize, int kSize) {
  dawn::float_type* reshaped = new dawn::float_type[field.numElements()];
  reshape(field.data(), reshaped, kSize, denseSize);
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * field.numElements()));
  gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * field.numElements(),
                       cudaMemcpyHostToDevice));
  delete[] reshaped;
}
void initField(const atlasInterface::Field<dawn::float_type>& field_x,
               const atlasInterface::Field<dawn::float_type>& field_y, float2** cudaStorage,
               int denseSize, int kSize) {
  assert(field_x.numElements() == field_y.numElements());
  dawn::float_type* reshaped_x = new dawn::float_type[field_x.numElements()];
  dawn::float_type* reshaped_y = new dawn::float_type[field_y.numElements()];
  reshape(field_x.data(), reshaped_x, kSize, denseSize);
  reshape(field_y.data(), reshaped_y, kSize, denseSize);

  float2* packed = new float2[field_x.numElements()];
  for(int i = 0; i < kSize * denseSize; i++) {
    packed[i] = make_float2(reshaped_x[i], reshaped_y[i]);
  }

  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(float2) * field_x.numElements()));
  gpuErrchk(cudaMemcpy(*cudaStorage, packed, sizeof(float2) * field_x.numElements(),
                       cudaMemcpyHostToDevice));
  delete[] reshaped_x;
  delete[] reshaped_y;
  delete[] packed;
}
void initSparseField(const atlasInterface::SparseDimension<dawn::float_type>& field_x,
                     const atlasInterface::SparseDimension<dawn::float_type>& field_y,
                     float2** cudaStorage, int denseSize, int sparseSize, int kSize) {
  assert(field_x.numElements() == field_y.numElements());
  dawn::float_type* reshaped_x = new dawn::float_type[field_x.numElements()];
  dawn::float_type* reshaped_y = new dawn::float_type[field_y.numElements()];
  reshape(field_x.data(), reshaped_x, kSize, denseSize, sparseSize);
  reshape(field_y.data(), reshaped_y, kSize, denseSize, sparseSize);

  float2* packed = new float2[field_x.numElements()];
  for(int i = 0; i < field_x.numElements(); i++) {
    packed[i] = make_float2(reshaped_x[i], reshaped_y[i]);
  }

  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(float2) * field_x.numElements()));
  gpuErrchk(cudaMemcpy(*cudaStorage, packed, sizeof(float2) * field_x.numElements(),
                       cudaMemcpyHostToDevice));
  delete[] reshaped_x;
  delete[] reshaped_y;
  delete[] packed;
}
void initSparseField(const atlasInterface::SparseDimension<dawn::float_type>& field,
                     dawn::float_type** cudaStorage, int denseSize, int sparseSize, int kSize) {
  dawn::float_type* reshaped = new dawn::float_type[field.numElements()];
  reshape(field.data(), reshaped, kSize, denseSize, sparseSize);
  gpuErrchk(cudaMalloc((void**)cudaStorage, sizeof(dawn::float_type) * field.numElements()));
  gpuErrchk(cudaMemcpy(*cudaStorage, reshaped, sizeof(dawn::float_type) * field.numElements(),
                       cudaMemcpyHostToDevice));
  delete[] reshaped;
}

DiamondStencil::diamond_stencil::diamond_stencil(
    const atlas::Mesh& mesh, int k_size,
    const atlasInterface::Field<dawn::float_type>& diff_multfac_smag,
    const atlasInterface::Field<dawn::float_type>& tangent_orientation,
    const atlasInterface::Field<dawn::float_type>& inv_primal_edge_length,
    const atlasInterface::Field<dawn::float_type>& inv_vert_vert_length,
    const atlasInterface::Field<dawn::float_type>& u_vert,
    const atlasInterface::Field<dawn::float_type>& v_vert,
    const atlasInterface::SparseDimension<dawn::float_type>& primal_normal_vert_x,
    const atlasInterface::SparseDimension<dawn::float_type>& primal_normal_vert_y,
    const atlasInterface::SparseDimension<dawn::float_type>& dual_normal_vert_x,
    const atlasInterface::SparseDimension<dawn::float_type>& dual_normal_vert_y,
    const atlasInterface::SparseDimension<dawn::float_type>& vn_vert,
    const atlasInterface::Field<dawn::float_type>& vn,
    const atlasInterface::Field<dawn::float_type>& dvt_tang,
    const atlasInterface::Field<dawn::float_type>& dvt_norm,
    const atlasInterface::Field<dawn::float_type>& kh_smag_1,
    const atlasInterface::Field<dawn::float_type>& kh_smag_2,
    const atlasInterface::Field<dawn::float_type>& kh_smag_e,
    const atlasInterface::Field<dawn::float_type>& z_nabla2_e)
    : sbase("diamond_stencil"), mesh_(mesh), kSize_(k_size) {
  initField(diff_multfac_smag, &diff_multfac_smag_, mesh.edges().size(), k_size);
  initField(tangent_orientation, &tangent_orientation_, mesh.edges().size(), k_size);
  initField(inv_primal_edge_length, &inv_primal_edge_length_, mesh.edges().size(), k_size);
  initField(inv_vert_vert_length, &inv_vert_vert_length_, mesh.edges().size(), k_size);
  initField(u_vert, v_vert, &uv_, mesh.nodes().size(), k_size);

  initSparseField(primal_normal_vert_x, primal_normal_vert_y, &primal_normal_vert_,
                  mesh.edges().size(), E_C_V_SIZE, k_size);
  initSparseField(dual_normal_vert_x, dual_normal_vert_y, &dual_normal_vert_, mesh.edges().size(),
                  E_C_V_SIZE, k_size);
  initSparseField(vn_vert, &vn_vert_, mesh.edges().size(), E_C_V_SIZE, k_size);

  initField(vn, &vn_, mesh.edges().size(), k_size);
  initField(dvt_tang, &dvt_tang_, mesh.edges().size(), k_size);
  initField(dvt_norm, &dvt_norm_, mesh.edges().size(), k_size);
  initField(kh_smag_1, &kh_smag_1_, mesh.edges().size(), k_size);
  initField(kh_smag_2, &kh_smag_2_, mesh.edges().size(), k_size);
  initField(kh_smag_e, &kh_smag_e_, mesh.edges().size(), k_size);
  initField(z_nabla2_e, &z_nabla2_e_, mesh.edges().size(), k_size);
}

void DiamondStencil::diamond_stencil::run() {
  int dK = (kSize_ + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
  dim3 dG((mesh_.NumEdges() + BLOCK_SIZE - 1) / BLOCK_SIZE, (dK + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  dim3 dB(BLOCK_SIZE, BLOCK_SIZE, 1);

  // starting timers
  start();

  merged<<<dG, dB>>>(mesh_.NumEdges(), mesh_.NumNodes(), kSize_, mesh_.ECVTable(), z_nabla2_e_,
                     inv_primal_edge_length_, inv_vert_vert_length_, tangent_orientation_, uv_,
                     primal_normal_vert_, dual_normal_vert_, vn_, diff_multfac_smag_, kh_smag_e_);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // stopping timers
  pause();
}

void DiamondStencil::diamond_stencil::CopyResultToHost(
    atlasInterface::Field<dawn::float_type>& kh_smag_e,
    atlasInterface::Field<dawn::float_type>& z_nabla2_e) const {
  gpuErrchk(cudaMemcpy((dawn::float_type*)kh_smag_e.data(), kh_smag_e_,
                       sizeof(dawn::float_type) * kh_smag_e.numElements(), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((dawn::float_type*)z_nabla2_e.data(), z_nabla2_e_,
                       sizeof(dawn::float_type) * z_nabla2_e.numElements(),
                       cudaMemcpyDeviceToHost));

  dawn::float_type* kh_smag_e_for_atlas = new dawn::float_type[kh_smag_e.numElements()];
  dawn::float_type* z_nabla2_e_for_atlas = new dawn::float_type[z_nabla2_e.numElements()];

  reshape_back(kh_smag_e.data(), kh_smag_e_for_atlas, kSize_, mesh_.NumEdges());
  reshape_back(z_nabla2_e.data(), z_nabla2_e_for_atlas, kSize_, mesh_.NumEdges());

  memcpy((dawn::float_type*)kh_smag_e.data(), kh_smag_e_for_atlas,
         sizeof(dawn::float_type) * kh_smag_e.numElements());
  memcpy((dawn::float_type*)z_nabla2_e.data(), z_nabla2_e_for_atlas,
         sizeof(dawn::float_type) * z_nabla2_e.numElements());
  delete[] kh_smag_e_for_atlas;
  delete[] z_nabla2_e_for_atlas;
}

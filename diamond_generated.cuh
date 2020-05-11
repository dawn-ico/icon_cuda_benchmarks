#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/math.hpp"
#include "driver-includes/timer_cuda.hpp"
#include "driver-includes/unstructured_interface.hpp"
#define BLOCK_SIZE 16
#define LEVELS_PER_THREAD 1
using namespace gridtools::dawn;

namespace dawn_generated {
namespace cuda_ico {
template <int E_C_V_SIZE>
__global__ void ICON_laplacian_diamond_stencil_stencil333_ms439_s466_kernel(
    int NumEdges, int NumVertices, int kSize, const int* ecvTable,
    const ::dawn::float_type* diff_multfac_smag, const ::dawn::float_type* tangent_orientation,
    const ::dawn::float_type* inv_primal_edge_length,
    const ::dawn::float_type* inv_vert_vert_length, const ::dawn::float_type* u_vert,
    const ::dawn::float_type* v_vert, const ::dawn::float_type* primal_normal_x,
    const ::dawn::float_type* primal_normal_y, const ::dawn::float_type* dual_normal_x,
    const ::dawn::float_type* dual_normal_y, ::dawn::float_type* vn_vert,
    const ::dawn::float_type* vn, ::dawn::float_type* dvt_tang, ::dawn::float_type* dvt_norm,
    ::dawn::float_type* kh_smag_1, ::dawn::float_type* kh_smag_2, ::dawn::float_type* kh_smag,
    ::dawn::float_type* nabla2) {
  unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int kidx = blockIdx.y * blockDim.y + threadIdx.y;
  int klo = kidx * LEVELS_PER_THREAD;
  int khi = (kidx + 1) * LEVELS_PER_THREAD;
  if(pidx >= NumEdges) {
    return;
  }
  for(int kIter = klo; kIter < khi; kIter++) {
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) {
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      vn_vert[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx] =
          ((u_vert[kIter * NumVertices + nbhIdx] *
            primal_normal_x[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx]) +
           (v_vert[kIter * NumVertices + nbhIdx] *
            primal_normal_y[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx]));
    }
    ::dawn::float_type lhs_495 = (::dawn::float_type)0.0;
    ::dawn::float_type weights_495[4] = {(::dawn::float_type)-1.0, (::dawn::float_type)1.0,
                                         (::dawn::float_type)0.0, (::dawn::float_type)0.0};
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) {
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs_495 += weights_495[nbhIter] *
                 ((u_vert[kIter * NumVertices + nbhIdx] *
                   dual_normal_x[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx]) +
                  (v_vert[kIter * NumVertices + nbhIdx] *
                   dual_normal_y[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx]));
    }
    dvt_tang[kIter * NumEdges + pidx] = lhs_495;
    dvt_tang[kIter * NumEdges + pidx] =
        (dvt_tang[kIter * NumEdges + pidx] * tangent_orientation[kIter * NumEdges + pidx]);
    ::dawn::float_type lhs_517 = (::dawn::float_type)0.0;
    ::dawn::float_type weights_517[4] = {(::dawn::float_type)0.0, (::dawn::float_type)0.0,
                                         (::dawn::float_type)-1.0, (::dawn::float_type)1.0};
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) {
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs_517 += weights_517[nbhIter] *
                 ((u_vert[kIter * NumVertices + nbhIdx] *
                   dual_normal_x[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx]) +
                  (v_vert[kIter * NumVertices + nbhIdx] *
                   dual_normal_y[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx]));
    }
    dvt_norm[kIter * NumEdges + pidx] = lhs_517;
    ::dawn::float_type lhs_527 = (::dawn::float_type)0.0;
    ::dawn::float_type weights_527[4] = {(::dawn::float_type)-1.0, (::dawn::float_type)1.0,
                                         (::dawn::float_type)0.0, (::dawn::float_type)0.0};
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) {
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs_527 +=
          weights_527[nbhIter] * vn_vert[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx];
    }
    kh_smag_1[kIter * NumEdges + pidx] = lhs_527;
    kh_smag_1[kIter * NumEdges + pidx] =
        (((kh_smag_1[kIter * NumEdges + pidx] * tangent_orientation[kIter * NumEdges + pidx]) *
          inv_primal_edge_length[kIter * NumEdges + pidx]) +
         (dvt_norm[kIter * NumEdges + pidx] * inv_vert_vert_length[kIter * NumEdges + pidx]));
    kh_smag_1[kIter * NumEdges + pidx] =
        (kh_smag_1[kIter * NumEdges + pidx] * kh_smag_1[kIter * NumEdges + pidx]);
    ::dawn::float_type lhs_555 = (::dawn::float_type)0.0;
    ::dawn::float_type weights_555[4] = {(::dawn::float_type)0.0, (::dawn::float_type)0.0,
                                         (::dawn::float_type)-1.0, (::dawn::float_type)1.0};
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) {
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs_555 +=
          weights_555[nbhIter] * vn_vert[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx];
    }
    kh_smag_2[kIter * NumEdges + pidx] = lhs_555;
    kh_smag_2[kIter * NumEdges + pidx] =
        ((kh_smag_2[kIter * NumEdges + pidx] * inv_vert_vert_length[kIter * NumEdges + pidx]) +
         (dvt_tang[kIter * NumEdges + pidx] * inv_primal_edge_length[kIter * NumEdges + pidx]));
    kh_smag_2[kIter * NumEdges + pidx] =
        (kh_smag_2[kIter * NumEdges + pidx] * kh_smag_2[kIter * NumEdges + pidx]);
    kh_smag[kIter * NumEdges + pidx] =
        (diff_multfac_smag[kIter * NumEdges + pidx] *
         sqrt((kh_smag_1[kIter * NumEdges + pidx] + kh_smag_2[kIter * NumEdges + pidx])));
    ::dawn::float_type lhs_600 = (::dawn::float_type)0.0;
    ::dawn::float_type weights_600[4] = {(inv_primal_edge_length[kIter * NumEdges + pidx] *
                                          inv_primal_edge_length[kIter * NumEdges + pidx]),
                                         (inv_primal_edge_length[kIter * NumEdges + pidx] *
                                          inv_primal_edge_length[kIter * NumEdges + pidx]),
                                         (inv_vert_vert_length[kIter * NumEdges + pidx] *
                                          inv_vert_vert_length[kIter * NumEdges + pidx]),
                                         (inv_vert_vert_length[kIter * NumEdges + pidx] *
                                          inv_vert_vert_length[kIter * NumEdges + pidx])};
    for(int nbhIter = 0; nbhIter < E_C_V_SIZE; nbhIter++) {
      int nbhIdx = __ldg(&ecvTable[pidx * E_C_V_SIZE + nbhIter]);
      if(nbhIdx == DEVICE_MISSING_VALUE) {
        continue;
      }
      lhs_600 += weights_600[nbhIter] *
                 ((::dawn::float_type)4.0 *
                  vn_vert[kIter * NumEdges * E_C_V_SIZE + nbhIter * NumEdges + pidx]);
    }
    nabla2[kIter * NumEdges + pidx] = lhs_600;
    nabla2[kIter * NumEdges + pidx] = (nabla2[kIter * NumEdges + pidx] -
                                       ((((::dawn::float_type)8.0 * vn[kIter * NumEdges + pidx]) *
                                         (inv_primal_edge_length[kIter * NumEdges + pidx] *
                                          inv_primal_edge_length[kIter * NumEdges + pidx])) +
                                        (((::dawn::float_type)8.0 * vn[kIter * NumEdges + pidx]) *
                                         (inv_vert_vert_length[kIter * NumEdges + pidx] *
                                          inv_vert_vert_length[kIter * NumEdges + pidx]))));
  }
}
template <typename LibTag, int E_C_V_SIZE>
class ICON_laplacian_diamond_stencil {
public:
  struct sbase : public timer_cuda {

    sbase(std::string name) : timer_cuda(name) {}

    double get_time() { return total_time(); }
  };

  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int* ecvTable;

    GpuTriMesh(const dawn::mesh_t<LibTag>& mesh) {
      NumVertices = mesh.nodes().size();
      NumCells = mesh.cells().size();
      NumEdges = mesh.edges().size();
      gpuErrchk(cudaMalloc((void**)&ecvTable, sizeof(int) * mesh.edges().size() * E_C_V_SIZE));
      dawn::generateNbhTable<LibTag>(
          mesh,
          {dawn::LocationType::Edges, dawn::LocationType::Cells, dawn::LocationType::Vertices},
          mesh.edges().size(), E_C_V_SIZE, ecvTable);
    }
  };

  struct stencil_333 : public sbase {
  private:
    dawn::float_type* diff_multfac_smag_;
    dawn::float_type* tangent_orientation_;
    dawn::float_type* inv_primal_edge_length_;
    dawn::float_type* inv_vert_vert_length_;
    dawn::float_type* u_vert_;
    dawn::float_type* v_vert_;
    dawn::float_type* primal_normal_x_;
    dawn::float_type* primal_normal_y_;
    dawn::float_type* dual_normal_x_;
    dawn::float_type* dual_normal_y_;
    dawn::float_type* vn_vert_;
    dawn::float_type* vn_;
    dawn::float_type* dvt_tang_;
    dawn::float_type* dvt_norm_;
    dawn::float_type* kh_smag_1_;
    dawn::float_type* kh_smag_2_;
    dawn::float_type* kh_smag_;
    dawn::float_type* nabla2_;
    int kSize_ = 0;
    GpuTriMesh mesh_;

  public:
    stencil_333(const dawn::mesh_t<LibTag>& mesh, int kSize,
                dawn::edge_field_t<LibTag, dawn::float_type>& diff_multfac_smag,
                dawn::edge_field_t<LibTag, dawn::float_type>& tangent_orientation,
                dawn::edge_field_t<LibTag, dawn::float_type>& inv_primal_edge_length,
                dawn::edge_field_t<LibTag, dawn::float_type>& inv_vert_vert_length,
                dawn::vertex_field_t<LibTag, dawn::float_type>& u_vert,
                dawn::vertex_field_t<LibTag, dawn::float_type>& v_vert,
                dawn::sparse_edge_field_t<LibTag, dawn::float_type>& primal_normal_x,
                dawn::sparse_edge_field_t<LibTag, dawn::float_type>& primal_normal_y,
                dawn::sparse_edge_field_t<LibTag, dawn::float_type>& dual_normal_x,
                dawn::sparse_edge_field_t<LibTag, dawn::float_type>& dual_normal_y,
                dawn::sparse_edge_field_t<LibTag, dawn::float_type>& vn_vert,
                dawn::edge_field_t<LibTag, dawn::float_type>& vn,
                dawn::edge_field_t<LibTag, dawn::float_type>& dvt_tang,
                dawn::edge_field_t<LibTag, dawn::float_type>& dvt_norm,
                dawn::edge_field_t<LibTag, dawn::float_type>& kh_smag_1,
                dawn::edge_field_t<LibTag, dawn::float_type>& kh_smag_2,
                dawn::edge_field_t<LibTag, dawn::float_type>& kh_smag,
                dawn::edge_field_t<LibTag, dawn::float_type>& nabla2)
        : sbase("stencil_333"), mesh_(mesh), kSize_(kSize) {
      dawn::initField(diff_multfac_smag, &diff_multfac_smag_, mesh.edges().size(), kSize);
      dawn::initField(tangent_orientation, &tangent_orientation_, mesh.edges().size(), kSize);
      dawn::initField(inv_primal_edge_length, &inv_primal_edge_length_, mesh.edges().size(), kSize);
      dawn::initField(inv_vert_vert_length, &inv_vert_vert_length_, mesh.edges().size(), kSize);
      dawn::initField(u_vert, &u_vert_, mesh.nodes().size(), kSize);
      dawn::initField(v_vert, &v_vert_, mesh.nodes().size(), kSize);
      dawn::initSparseField(primal_normal_x, &primal_normal_x_, mesh.edges().size(), E_C_V_SIZE,
                            kSize);
      dawn::initSparseField(primal_normal_y, &primal_normal_y_, mesh.edges().size(), E_C_V_SIZE,
                            kSize);
      dawn::initSparseField(dual_normal_x, &dual_normal_x_, mesh.edges().size(), E_C_V_SIZE, kSize);
      dawn::initSparseField(dual_normal_y, &dual_normal_y_, mesh.edges().size(), E_C_V_SIZE, kSize);
      dawn::initSparseField(vn_vert, &vn_vert_, mesh.edges().size(), E_C_V_SIZE, kSize);
      dawn::initField(vn, &vn_, mesh.edges().size(), kSize);
      dawn::initField(dvt_tang, &dvt_tang_, mesh.edges().size(), kSize);
      dawn::initField(dvt_norm, &dvt_norm_, mesh.edges().size(), kSize);
      dawn::initField(kh_smag_1, &kh_smag_1_, mesh.edges().size(), kSize);
      dawn::initField(kh_smag_2, &kh_smag_2_, mesh.edges().size(), kSize);
      dawn::initField(kh_smag, &kh_smag_, mesh.edges().size(), kSize);
      dawn::initField(nabla2, &nabla2_, mesh.edges().size(), kSize);
    }

    void run() {
      int dK = (kSize_ + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
      dim3 dGE((mesh_.NumEdges + BLOCK_SIZE - 1) / BLOCK_SIZE, (dK + BLOCK_SIZE - 1) / BLOCK_SIZE,
               1);
      dim3 dB(BLOCK_SIZE, BLOCK_SIZE, 1);
      sbase::start();
      ICON_laplacian_diamond_stencil_stencil333_ms439_s466_kernel<E_C_V_SIZE><<<dGE, dB>>>(
          mesh_.NumEdges, mesh_.NumVertices, kSize_, mesh_.ecvTable, diff_multfac_smag_,
          tangent_orientation_, inv_primal_edge_length_, inv_vert_vert_length_, u_vert_, v_vert_,
          primal_normal_x_, primal_normal_y_, dual_normal_x_, dual_normal_y_, vn_vert_, vn_,
          dvt_tang_, dvt_norm_, kh_smag_1_, kh_smag_2_, kh_smag_, nabla2_);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      sbase::pause();
    }

    void CopyResultToHost(dawn::sparse_edge_field_t<LibTag, dawn::float_type>& vn_vert,
                          dawn::edge_field_t<LibTag, dawn::float_type>& dvt_tang,
                          dawn::edge_field_t<LibTag, dawn::float_type>& dvt_norm,
                          dawn::edge_field_t<LibTag, dawn::float_type>& kh_smag_1,
                          dawn::edge_field_t<LibTag, dawn::float_type>& kh_smag_2,
                          dawn::edge_field_t<LibTag, dawn::float_type>& kh_smag,
                          dawn::edge_field_t<LibTag, dawn::float_type>& nabla2) {
      {
        dawn::float_type* host_buf = new dawn::float_type[vn_vert.numElements()];
        gpuErrchk(cudaMemcpy((dawn::float_type*)host_buf, vn_vert_,
                             vn_vert.numElements() * sizeof(dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, vn_vert.data(), kSize_, mesh_.NumEdges, E_C_V_SIZE);
        delete[] host_buf;
      }
      {
        dawn::float_type* host_buf = new dawn::float_type[dvt_tang.numElements()];
        gpuErrchk(cudaMemcpy((dawn::float_type*)host_buf, dvt_tang_,
                             dvt_tang.numElements() * sizeof(dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, dvt_tang.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      {
        dawn::float_type* host_buf = new dawn::float_type[dvt_norm.numElements()];
        gpuErrchk(cudaMemcpy((dawn::float_type*)host_buf, dvt_norm_,
                             dvt_norm.numElements() * sizeof(dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, dvt_norm.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      {
        dawn::float_type* host_buf = new dawn::float_type[kh_smag_1.numElements()];
        gpuErrchk(cudaMemcpy((dawn::float_type*)host_buf, kh_smag_1_,
                             kh_smag_1.numElements() * sizeof(dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, kh_smag_1.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      {
        dawn::float_type* host_buf = new dawn::float_type[kh_smag_2.numElements()];
        gpuErrchk(cudaMemcpy((dawn::float_type*)host_buf, kh_smag_2_,
                             kh_smag_2.numElements() * sizeof(dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, kh_smag_2.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      {
        dawn::float_type* host_buf = new dawn::float_type[kh_smag.numElements()];
        gpuErrchk(cudaMemcpy((dawn::float_type*)host_buf, kh_smag_,
                             kh_smag.numElements() * sizeof(dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, kh_smag.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
      {
        dawn::float_type* host_buf = new dawn::float_type[nabla2.numElements()];
        gpuErrchk(cudaMemcpy((dawn::float_type*)host_buf, nabla2_,
                             nabla2.numElements() * sizeof(dawn::float_type),
                             cudaMemcpyDeviceToHost));
        dawn::reshape_back(host_buf, nabla2.data(), kSize_, mesh_.NumEdges);
        delete[] host_buf;
      }
    }
  };
};
} // namespace cuda_ico
} // namespace dawn_generated

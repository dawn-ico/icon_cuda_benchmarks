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

#pragma once

#include "vector_types.h"

#include <atlas/array.h>
#include <atlas/mesh.h>

#include "atlas_interface.hpp"

#include "timer_cuda.hpp"

#include <cuda.h>

class GpuTriMesh {
private:
  int* ecvTable_ = 0;

  int numNodes_ = 0;
  int numEdges_ = 0;
  int numCells_ = 0;

public:
  int NumNodes() const { return numNodes_; }
  int NumEdges() const { return numEdges_; }
  int NumCells() const { return numCells_; }

  int* ECVTable() const { return ecvTable_; }

  GpuTriMesh(const atlas::Mesh& mesh);
};

class DiamondStencil {
public:
  struct sbase : public gridtools::dawn::timer_cuda {
    sbase(std::string name) : timer_cuda(name) {}

    dawn::float_type get_time() { return total_time(); }
  };

  struct diamond_stencil : public sbase {

  private:
    // --------------------------
    // fields
    // --------------------------
    // in dense
    dawn::float_type* diff_multfac_smag_;
    dawn::float_type* tangent_orientation_;
    dawn::float_type* inv_primal_edge_length_;
    dawn::float_type* inv_vert_vert_length_;
    // dawn::float_type* u_vert_;
    // dawn::float_type* v_vert_;
    dawn::float_2_type* uv_;

    // in sparse
    // dawn::float_type* primal_normal_vert_x_;
    // dawn::float_type* primal_normal_vert_y_;
    // dawn::float_type* dual_normal_vert_x_;
    // dawn::float_type* dual_normal_vert_y_;
    dawn::float_2_type* primal_normal_vert_;
    dawn::float_2_type* dual_normal_vert_;

    // input field on vertices and eges
    dawn::float_type* vn_vert_;
    dawn::float_type* vn_;

    // tangential and normal coefficient for smagorinsky
    dawn::float_type* dvt_tang_;
    dawn::float_type* dvt_norm_;

    // out
    dawn::float_type* kh_smag_1_;
    dawn::float_type* kh_smag_2_;
    dawn::float_type* kh_smag_e_;
    dawn::float_type* z_nabla2_e_;

    // --------------------------
    // number of levels per field
    // --------------------------
    int kSize_ = 0;

    // --------------------------
    // mesh
    // --------------------------
    GpuTriMesh mesh_;

  public:
    diamond_stencil(const atlas::Mesh& mesh, int k_size,
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
                    const atlasInterface::Field<dawn::float_type>& z_nabla2_e);
    void run();

    void CopyResultToHost(atlasInterface::Field<dawn::float_type>& kh_smag_e,
                          atlasInterface::Field<dawn::float_type>& z_nabla2_e) const;
  };
};
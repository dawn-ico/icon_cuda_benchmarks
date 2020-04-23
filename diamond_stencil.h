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

class GpuTriMesh
{
private:
  double2 *pos_ = 0;
  int *nodeToCell_ = 0;
  int *nodeToEdge_ = 0;
  int *edgeToCell_ = 0;
  int *edgeToNode_ = 0;
  int *cellToNode_ = 0;
  int *cellToEdge_ = 0;
  int *ecvTable_ = 0;

  int numNodes_ = 0;
  int numEdges_ = 0;
  int numCells_ = 0;

public:
  int NumNodes() const { return numNodes_; }
  int NumEdges() const { return numEdges_; }
  int NumCells() const { return numCells_; }

  double2 *Pos() const { return pos_; };
  int *NodeToCell() const { return nodeToCell_; };
  int *NodeToEdge() const { return nodeToEdge_; };
  int *EdgeToCell() const { return edgeToCell_; };
  int *EdgeToNode() const { return edgeToNode_; };
  int *CellToNode() const { return cellToNode_; };
  int *CellToEdge() const { return cellToEdge_; };
  int *ECVTable() const { return ecvTable_; }

  GpuTriMesh(const atlas::Mesh &mesh);
};

class DiamondStencil
{
public:
  struct sbase : public gridtools::dawn::timer_cuda
  {
    sbase(std::string name) : timer_cuda(name) {}

    double get_time() { return total_time(); }
  };

  struct diamond_stencil : public sbase
  {

  private:
    // --------------------------
    // fields
    // --------------------------
    // in dense
    double *diff_multfac_smag_;
    double *u_vert_;
    double *v_vert_;
    double *tangent_orientation_;
    double *inv_primal_edge_length_;
    double *inv_vert_vert_length_;
    double *dvt_tang_;
    double *dvt_norm_;

    // in sparse
    double *vn_vert_;
    double *primal_normal_vert_x_;
    double *primal_normal_vert_y_;
    double *dual_normal_vert_x_;
    double *dual_normal_vert_y_;

    // out
    double *kh_smag_e_;
    double *kh_smag_ec_;
    double *z_nabla2_e_;

    // --------------------------
    // mesh
    // --------------------------
    GpuTriMesh mesh_;

  public:
    diamond_stencil(const atlas::Mesh &mesh,
                    const atlasInterface::Field<double> &diff_multfac_smag,
                    const atlasInterface::Field<double> &u_vert,
                    const atlasInterface::Field<double> &v_vert,
                    const atlasInterface::Field<double> &tangent_orientation,
                    const atlasInterface::Field<double> &inv_primal_edge_length,
                    const atlasInterface::Field<double> &inv_vert_vert_length,
                    const atlasInterface::Field<double> &dvt_tang,
                    const atlasInterface::Field<double> &dvt_norm,
                    const atlasInterface::SparseDimension<double> &vn_vert,
                    const atlasInterface::SparseDimension<double> &primal_normal_vert_x,
                    const atlasInterface::SparseDimension<double> &primal_normal_vert_y,
                    const atlasInterface::SparseDimension<double> &dual_normal_vert_x,
                    const atlasInterface::SparseDimension<double> &dual_normal_vert_y,
                    const atlasInterface::Field<double> &kh_smag_e,
                    const atlasInterface::Field<double> &kh_smag_ec,
                    const atlasInterface::Field<double> &z_nabla2_e);
    void run();

    void CopyResultToHost(atlasInterface::Field<double> &kh_smag_e,
                          atlasInterface::Field<double> &kh_smag_ec,
                          atlasInterface::Field<double> &z_nabla2_e) const;
  };
};
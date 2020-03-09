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
  double2* pos_ = 0;
  int* nodeToCell_ = 0;
  int* nodeToEdge_ = 0;
  int* edgeToCell_ = 0;
  int* edgeToNode_ = 0;
  int* cellToNode_ = 0;
  int* cellToEdge_ = 0;

  int numNodes_ = 0;
  int numEdges_ = 0;
  int numCells_ = 0;

public:
  int NumNodes() const { return numNodes_; }
  int NumEdges() const { return numEdges_; }
  int NumCells() const { return numCells_; }

  double2* Pos() const { return pos_; };
  int* NodeToCell() const { return nodeToCell_; };
  int* NodeToEdge() const { return nodeToEdge_; };
  int* EdgeToCell() const { return edgeToCell_; };
  int* EdgeToNode() const { return edgeToNode_; };
  int* CellToNode() const { return cellToNode_; };
  int* CellToEdge() const { return cellToEdge_; };

  GpuTriMesh(const atlas::Mesh& mesh);
};

class LaplacianStencil {
public:
  struct sbase : public gridtools::dawn::timer_cuda {
    sbase(std::string name) : timer_cuda(name) {}

    double get_time() { return total_time(); }
  };

  struct laplacian_stencil : public sbase {

  private:
    // --------------------------
    // fields
    // --------------------------
    // in dense
    double* vec_;
    double* primal_edge_length_;
    double* dual_edge_length_;
    double* tangent_orientation_;

    // in sparse
    double* geofacDiv_;
    double* geofacRot_;

    // out
    double* divVec_;
    double* rotVec_;
    double* nabla2t1vec_;
    double* nabla2t2vec_;
    double* nabla2vec_;

    // --------------------------
    // mesh
    // --------------------------
    GpuTriMesh mesh_;

  public:
    laplacian_stencil(const atlas::Mesh& mesh, const atlasInterface::Field<double>& vec,
                      const atlasInterface::Field<double>& rotVec,
                      const atlasInterface::SparseDimension<double>& geofacRot,
                      const atlasInterface::Field<double>& divVec,
                      const atlasInterface::SparseDimension<double>& geofacDiv,
                      const atlasInterface::Field<double>& primal_edge_length,
                      const atlasInterface::Field<double>& dual_edge_length,
                      const atlasInterface::Field<double>& tangent_orientation,
                      const atlasInterface::Field<double>& nabla2t1vec,
                      const atlasInterface::Field<double>& nabla2t2vec,
                      const atlasInterface::Field<double>& nabla2vec);
    void run();

    void CopyResultToHost(atlasInterface::Field<double>& rotVec,
                          atlasInterface::Field<double>& divVec,
                          atlasInterface::Field<double>& nabla2t1vec,
                          atlasInterface::Field<double>& nabla2t2vec,
                          atlasInterface::Field<double>& nabla2vec) const;
  };
};
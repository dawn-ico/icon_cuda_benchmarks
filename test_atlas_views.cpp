
#include <cmath>
#include <cstdio>
#include <fenv.h>
#include <optional>
#include <vector>

// atlas functions
#include <atlas/array.h>
#include <atlas/grid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
// #include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/util/CoordinateEnums.h>

// atlas interface for dawn generated code
#include "atlas_interface.hpp"

#define initField(field, cudaStorage)                                                              \
  {                                                                                                \
    cudaStorage = (double*)malloc(sizeof(double) * field.numElements());                           \
    memcpy(cudaStorage, field.data(), sizeof(double) * field.numElements());                       \
  }

void cudakernel(int numEdges, int sparseSize, int kLevels, double* ptr, int pid) {
  for(int sparseIdx = 0; sparseIdx < sparseSize; sparseIdx++) {
    for(int kIter = 0; kIter < kLevels; kIter++) {
      const int ecvSparseKOffset = kIter * numEdges * sparseSize;

      //(elem_idx, level, sparse_dim_idx); <- atlas addressing
      // printf("%03d\n", (int)ptr[ecvSparseKOffset + pid * sparseSize + sparseIdx]);

      // new theory
      // edges, k, sparse
      printf("%03d\n", (int)ptr[pid * kLevels * sparseSize + kIter * sparseSize + sparseIdx]);
    }
  }
}

int main() {
  int k_size = 3;
  int numEdges = 10;
  int numSparse = 4;

  auto MakeAtlasSparseField =
      [&](const std::string& name, int size,
          int sparseSize) -> std::tuple<atlas::Field, atlasInterface::SparseDimension<double>> {
    atlas::Field field_F{name, atlas::array::DataType::real64(),
                         atlas::array::make_shape(numEdges, k_size, sparseSize)};
    return {field_F, atlas::array::make_view<double, 3>(field_F)};
  };

  auto [field, view] = MakeAtlasSparseField("test", numEdges, numSparse);

  for(int level = 0; level < k_size; level++) {
    for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
      for(int sparseIdx = 0; sparseIdx < numSparse; sparseIdx++) {
        std::string myStr =
            (std::to_string(level) + std::to_string(edgeIdx) + std::to_string(sparseIdx));
        view(edgeIdx, sparseIdx, level) = std::atoi(myStr.c_str());
      }
    }
  }

  double* cudaPtr;
  initField(view, cudaPtr);

  for(int edgeIdx = 0; edgeIdx < numEdges; edgeIdx++) {
    cudakernel(numEdges, numSparse, k_size, cudaPtr, edgeIdx);
  }

  return 0;
}
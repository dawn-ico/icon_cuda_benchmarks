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
#include <atlas/output/Gmsh.h>
#include <atlas/util/CoordinateEnums.h>

// atlas interface for dawn generated code
#include "atlas_interface.hpp"

// icon stencil
#include "diamond_stencil.h"

// atlas utilities
#include "atlas_utils/utils/AtlasCartesianWrapper.h"
#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"
#include "atlas_utils/utils/GenerateStrIndxAtlasMesh.h"

// io
#include "atlas_utils/stencils/io/atlasIO.h"

#include "driver-includes/defs.hpp"

std::string getCmdOption(const char** begin, const char** end, const std::string& option) {
  const char** itr = std::find(begin, end, option);
  if(itr != end && ++itr != end) {
    return *itr;
  }
  throw std::runtime_error("error, option :" + option + " not specified");

  return 0;
}

bool cmdOptionExists(const char** begin, const char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}

void printHelp(std::string program) {
  std::cout << program << " [options]" << std::endl;
  std::cout << "required arguments:" << std::endl;
  std::cout << " -ny <ny>          : number of rows for a square domain" << std::endl;
  std::cout << "optional arguments:" << std::endl;
  std::cout << " -str              : structured indexing layout" << std::endl;
  std::cout << " -d                 : debug " << std::endl;
}
std::tuple<dawn::float_type, dawn::float_type, dawn::float_type>
MeasureErrors(std::vector<int> indices, const atlasInterface::Field<dawn::float_type>& ref,
              const atlasInterface::Field<dawn::float_type>& sol, int k_size);

std::tuple<dawn::float_type, dawn::float_type, dawn::float_type>
MeasureErrors(const std::string& refFname, const atlasInterface::Field<dawn::float_type>& sol,
              int num_edges, int k_size);
std::tuple<dawn::float_type, dawn::float_type, dawn::float_type>
MeasureErrors(std::vector<int> indices, const std::vector<dawn::float_type>& ref,
              const atlasInterface::Field<dawn::float_type>& sol, int k_size);

int main(int argc, char const* argv[]) {
  // enable floating point exception
  // feenableexcept(FE_INVALID | FE_OVERFLOW);

  if(!cmdOptionExists(argv, argv + argc, "-ny")) {
    printHelp(argv[0]);
    return -1;
  }

  bool strLayout = false;
  if(cmdOptionExists(argv, argv + argc, "-str")) {
    strLayout = true;
  }

  int ny = std::stoi(getCmdOption(argv, argv + argc, "-ny"));
  int nx = std::stoi(getCmdOption(argv, argv + argc, "-nx"));

  bool debug = cmdOptionExists(argv, argv + argc, "-d") ? true : false;

  int k_size = 80;
  dawn::float_type lDomain = M_PI;

  time_t tic = clock();

  // dump a whole bunch of debug output (meant to be visualized using Octave, but gnuplot and the
  // like will certainly work too)
  const bool dbg_out = false;
  const bool verbose = true;
  const bool readMeshFromDisk = true;

  time_t meshgen_tic = clock();

  atlas::Mesh mesh;
  if(!readMeshFromDisk) {
    if(strLayout) {
      mesh = AtlasStrIndxMesh(nx, ny);
    } else {
      mesh = AtlasMeshRect(nx, ny);
    }
  } else {
    mesh = *AtlasMeshFromNetCDFComplete("grid.nc");
    auto lonlat = atlas::array::make_view<double, 2>(mesh.nodes().lonlat());
    auto xy = atlas::array::make_view<double, 2>(mesh.nodes().xy());

    auto lonToRad = [](double rad) { return rad * (0.5 * M_PI) / 90; };
    auto latToRad = [](double rad) { return rad * (M_PI) / 180; };

    for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
      xy(nodeIdx, atlas::LON) = lonToRad(lonlat(nodeIdx, atlas::LON));
      xy(nodeIdx, atlas::LAT) = latToRad(lonlat(nodeIdx, atlas::LAT));
    }
  }
  if(debug) {
    atlas::output::Gmsh gmsh("mesh.msh");
    gmsh.write(mesh);
  }

  time_t meshgen_toc = clock();

  if(verbose) {
    std::cout << "mesh generation took " << (meshgen_toc - meshgen_tic) / ((double)CLOCKS_PER_SEC)
              << " seconds \n";
  }

  // wrapper with various atlas helper functions
  AtlasToCartesian wrapper(mesh, !readMeshFromDisk);

  const int edgesPerVertex = 6;
  const int edgesPerCell = 3;
  const int verticesInDiamond = 4;

  // dumpMesh4Triplot(mesh, "atlas", wrapper);

  //===------------------------------------------------------------------------------------------===//
  // helper lambdas to readily construct atlas fields and views on one line
  //===------------------------------------------------------------------------------------------===//
  auto MakeAtlasField =
      [&](const std::string& name,
          int size) -> std::tuple<atlas::Field, atlasInterface::Field<dawn::float_type>> {
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real32(),
                         atlas::array::make_shape(size, k_size)};
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real64(),
                         atlas::array::make_shape(size, k_size)};
#else
#error DAWN_PRECISION is invalid
#endif
    return {field_F, atlas::array::make_view<dawn::float_type, 2>(field_F)};
  };

  auto MakeAtlasSparseField = [&](const std::string& name, int size, int sparseSize)
      -> std::tuple<atlas::Field, atlasInterface::SparseDimension<dawn::float_type>> {
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real32(),
                         atlas::array::make_shape(size, k_size, sparseSize)};
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real64(),
                         atlas::array::make_shape(size, k_size, sparseSize)};
#else
#error DAWN_PRECISION is invalid
#endif
    return {field_F, atlas::array::make_view<dawn::float_type, 3>(field_F)};
  };

  time_t alloc_tic = clock();

  //===------------------------------------------------------------------------------------------===//
  // input field (field we want to take the laplacian of)
  //  in the ICON stencil the velocity is reconstructed at vertices (from edges)
  //  for this test, we simply assign an analytical function
  //===------------------------------------------------------------------------------------------===//
  auto [u_F, u] = MakeAtlasField("u", mesh.nodes().size());
  auto [v_F, v] = MakeAtlasField("v", mesh.nodes().size());

  //===------------------------------------------------------------------------------------------===//
  // input field (field we want to take the laplacian of)
  //  normal velocity on edges
  //===------------------------------------------------------------------------------------------===//
  auto [vn_F, vn] = MakeAtlasField("vn", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // output fields (kh_smag(1|2) are "helper" fields to store intermediary results)
  //===------------------------------------------------------------------------------------------===//
  auto [nabla2_F, nabla2] = MakeAtlasField("nabla2", mesh.edges().size());
  auto [kh_smag_1_F, kh_smag_1] = MakeAtlasField("kh_smag_1", mesh.edges().size());
  auto [kh_smag_2_F, kh_smag_2] = MakeAtlasField("kh_smag_2", mesh.edges().size());
  auto [kh_smag_F, kh_smag] = MakeAtlasField("kh_smag", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // control field
  //===------------------------------------------------------------------------------------------===//
  auto [nabla2_sol_F, nabla2_sol] = MakeAtlasField("nabla2", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // geometrical quantities on edges (vert_vert_lenght is distance between far vertices of diamond)
  //===------------------------------------------------------------------------------------------===//
  auto [inv_primal_edge_length_F, inv_primal_edge_length] =
      MakeAtlasField("inv_primal_edge_length", mesh.edges().size());
  auto [inv_vert_vert_length_F, inv_vert_vert_length] =
      MakeAtlasField("inv_vert_vert_length", mesh.edges().size());
  auto [tangent_orientation_F, tangent_orientation] =
      MakeAtlasField("tangent_orientation", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // smagorinsky coefficient stored on edges (=1 for us, simply there to force the same number of
  // reads in both ICON and our version)
  //===------------------------------------------------------------------------------------------===//
  auto [diff_multfac_smag_F, diff_multfac_smag] =
      MakeAtlasField("diff_multfac_smag", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // tangential and normal components for smagorinsky diffusion
  //===------------------------------------------------------------------------------------------===//
  auto [dvt_norm_F, dvt_norm] = MakeAtlasField("dvt_norm", mesh.edges().size());
  auto [dvt_tang_F, dvt_tang] = MakeAtlasField("dvt_tang", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // primal and dual normals at vertices (!)
  //  supposedly simply a copy of the edge normal in planar geometry (to be checked)
  //===------------------------------------------------------------------------------------------===//
  auto [primal_normal_x_F, primal_normal_x] =
      MakeAtlasSparseField("primal_normal_x", mesh.edges().size(), verticesInDiamond);
  auto [primal_normal_y_F, primal_normal_y] =
      MakeAtlasSparseField("primal_normal_y", mesh.edges().size(), verticesInDiamond);
  auto [dual_normal_x_F, dual_normal_x] =
      MakeAtlasSparseField("dual_normal_x", mesh.edges().size(), verticesInDiamond);
  auto [dual_normal_y_F, dual_normal_y] =
      MakeAtlasSparseField("dual_normal_y", mesh.edges().size(), verticesInDiamond);

  //===------------------------------------------------------------------------------------------===//
  // sparse dimension intermediary field for diamond
  //===------------------------------------------------------------------------------------------===//
  auto [vn_vert_F, vn_vert] =
      MakeAtlasSparseField("vn_vert", mesh.edges().size(), verticesInDiamond);

  time_t alloc_toc = clock();

  if(verbose) {
    std::cout << "allocating fields took " << (alloc_toc - alloc_tic) / ((double)CLOCKS_PER_SEC)
              << " seconds \n";
  }

  //===------------------------------------------------------------------------------------------===//
  // input (spherical harmonics) and analytical solutions for div, curl and Laplacian
  //===------------------------------------------------------------------------------------------===//

  auto sphericalHarmonic =
      [](dawn::float_type x, dawn::float_type y) -> std::tuple<dawn::float_type, dawn::float_type> {
    dawn::float_type c1 = 0.25 * sqrt(105. / (2 * M_PI));
    dawn::float_type c2 = 0.5 * sqrt(15. / (2 * M_PI));
    return {c1 * cos(2 * x) * cos(y) * cos(y) * sin(y), c2 * cos(x) * cos(y) * sin(y)};
  };
  auto analyticalLaplacian =
      [](dawn::float_type x, dawn::float_type y) -> std::tuple<dawn::float_type, dawn::float_type> {
    dawn::float_type c1 = 0.25 * sqrt(105. / (2 * M_PI));
    dawn::float_type c2 = 0.5 * sqrt(15. / (2 * M_PI));
    return {-4 * c1 * cos(2 * x) * cos(y) * cos(y) * sin(y), -4 * c2 * cos(x) * sin(y) * cos(y)};
  };
  auto analyticalScalarLaplacian = [](dawn::float_type x, dawn::float_type y) -> dawn::float_type {
    dawn::float_type c1 = 0.25 * sqrt(105. / (2 * M_PI));
    dawn::float_type c2 = 0.5 * sqrt(15. / (2 * M_PI));
    return sin(y) * (-1. / 2. * c1 * cos(2 * x) * (13 * cos(2 * y) + 9) - 5 * c2 * cos(x) * cos(y));
  };

  auto wave = [](dawn::float_type x, dawn::float_type y) -> dawn::float_type {
    return sin(x) * sin(y);
  };
  auto analyticalLaplacianWave = [](dawn::float_type x, dawn::float_type y) -> dawn::float_type {
    return -2 * sin(x) * sin(y);
  };

  clock_t inout_tic = clock();

  std::vector<std::tuple<dawn::float_type, dawn::float_type>> primal_normal_cache(
      mesh.edges().size());

  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    primal_normal_cache[edgeIdx] = wrapper.primalNormal(mesh, edgeIdx);
  }

  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    auto [x, y] = wrapper.nodeLocation(nodeIdx);
    auto [ui, vi] = sphericalHarmonic(x, y);
    for(int level = 0; level < k_size; level++) {
      u(nodeIdx, level) = ui;
      v(nodeIdx, level) = vi;
    }
  }
  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    auto [nx, ny] = primal_normal_cache[edgeIdx];
    auto [x, y] = wrapper.edgeMidpoint(mesh, edgeIdx);
    auto [ui, vi] = sphericalHarmonic(x, y);
    for(int level = 0; level < k_size; level++) {
      // vn(edgeIdx, level) = ui * nx + vi * ny;
      vn(edgeIdx, level) = ui * 1. + vi * 1.;
    }
  }
  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    auto [x, y] = wrapper.edgeMidpoint(mesh, edgeIdx);
    auto [ui, vi] = analyticalLaplacian(x, y);
    auto [nx, ny] = primal_normal_cache[edgeIdx];
    double analytical = analyticalScalarLaplacian(x, y);
    for(int level = 0; level < k_size; level++) {
      nabla2_sol(edgeIdx, level) = analytical;
      // nabla2_sol(edgeIdx, level) = ui * nx + vi * ny;
    }
  }

  clock_t inout_toc = clock();

  if(verbose) {
    std::cout << "computing input and ref. solution took "
              << (inout_toc - inout_tic) / ((double)CLOCKS_PER_SEC) << " seconds \n";
  }

  clock_t geom_tic = clock();

  //===------------------------------------------------------------------------------------------===//
  // initialize geometrical info on edges
  //===------------------------------------------------------------------------------------------===//
  if(!readMeshFromDisk) {
    for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      double edgeLength = wrapper.edgeLength(mesh, edgeIdx);
      double tangentOrientation = wrapper.tangentOrientation(mesh, edgeIdx);
      dawn::float_type vert_vert_length = sqrt(3.) * edgeLength;
      for(int level = 0; level < k_size; level++) {
        inv_primal_edge_length(edgeIdx, level) = 1. / edgeLength;
        inv_vert_vert_length(edgeIdx, level) = (vert_vert_length == 0) ? 0 : 1. / vert_vert_length;
        tangent_orientation(edgeIdx, level) = tangentOrientation;
      }
    }
  } else {
    for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      double edgeLength = wrapper.edgeLength(mesh, edgeIdx);
      double tangentOrientation = wrapper.tangentOrientation(mesh, edgeIdx);
      dawn::float_type vert_vert_length = wrapper.vertVertLength(mesh, edgeIdx);
      for(int level = 0; level < k_size; level++) {
        inv_primal_edge_length(edgeIdx, level) = 1. / edgeLength;
        inv_vert_vert_length(edgeIdx, level) = (vert_vert_length == 0) ? 0 : 1. / vert_vert_length;
        tangent_orientation(edgeIdx, level) = tangentOrientation;
      }
    }
  }

  clock_t geom_toc = clock();

  if(verbose) {
    std::cout << "geometric info on edges took " << (geom_toc - geom_tic) / ((double)CLOCKS_PER_SEC)
              << " seconds \n";
  }

  //===------------------------------------------------------------------------------------------===//
  // initialize sparse geometrical info
  //===------------------------------------------------------------------------------------------===//
  for(int level = 0; level < k_size; level++) {
    for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      auto [nx, ny] = primal_normal_cache[edgeIdx];

      for(int nbhIdx = 0; nbhIdx < verticesInDiamond; nbhIdx++) {
        primal_normal_x(edgeIdx, nbhIdx, level) = 1.;
        primal_normal_y(edgeIdx, nbhIdx, level) = 1.;
        dual_normal_x(edgeIdx, nbhIdx, level) = 1.;
        dual_normal_y(edgeIdx, nbhIdx, level) = 1.;

        // primal_normal_x(edgeIdx, nbhIdx, level) = nx;
        // primal_normal_y(edgeIdx, nbhIdx, level) = ny;
        // dual_normal_x(edgeIdx, nbhIdx, level) = ny;
        // dual_normal_y(edgeIdx, nbhIdx, level) = -nx;
      }
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // smagorinsky fac (dummy)
  //===------------------------------------------------------------------------------------------===//
  for(int level = 0; level < k_size; level++) {
    for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      diff_multfac_smag(edgeIdx, level) = 1.;
    }
  }

  time_t toc = clock();

  if(verbose) {
    std::cout << "initialization took " << (toc - tic) / ((double)CLOCKS_PER_SEC) << " seconds \n";
  }

  //===------------------------------------------------------------------------------------------===//
  // stencil call
  //===------------------------------------------------------------------------------------------===/
  DiamondStencil::diamond_stencil lapl(
      mesh, k_size, diff_multfac_smag, tangent_orientation, inv_primal_edge_length,
      inv_vert_vert_length, u, v, primal_normal_x, primal_normal_y, dual_normal_x, dual_normal_y,
      vn_vert, vn, dvt_tang, dvt_norm, kh_smag_1, kh_smag_2, kh_smag, nabla2);

  const int warmup_runs = 100;
  for(int i = 0; i < warmup_runs; i++) {
    lapl.run();
    lapl.reset();
  }

  const int nruns = 100;
  std::vector<dawn::float_type> times(nruns);
  for(int i = 0; i < nruns; i++) {
    lapl.run();
    times[i] = lapl.get_time();
    lapl.reset();
  }
  lapl.CopyResultToHost(kh_smag, nabla2);

  auto mean = [](const std::vector<dawn::float_type>& times) {
    dawn::float_type avg = 0.;
    for(auto time : times) {
      avg += time;
    }
    return avg / times.size();
  };
  auto standard_deviation = [&](const std::vector<dawn::float_type>& times) {
    auto avg = mean(times);
    dawn::float_type sd = 0.;
    for(auto time : times) {
      sd += (time - avg) * (time - avg);
    }
    return sqrt(1. / (times.size() - 1) * sd);
  };

  std::cout << "average time for " << nruns << " run(s) of Laplacian: " << mean(times)
            << " with standard deviation of: " << standard_deviation(times) << "\n";

  //===------------------------------------------------------------------------------------------===//
  // dumping a hopefully nice colorful laplacian
  //===------------------------------------------------------------------------------------------===//
  // dumpEdgeField("diamondLaplICONatlas_out0.txt", mesh, wrapper, nabla2, 0,
  //               wrapper.innerEdges(mesh));
  // dumpEdgeField("diamondLaplICONatlas_out1.txt", mesh, wrapper, nabla2, 1,
  //               wrapper.innerEdges(mesh));
  // dumpEdgeField("diamondLaplICONatlas_sol.txt", mesh, wrapper, nabla2_sol, 0,
  //               wrapper.innerEdges(mesh));

  //===------------------------------------------------------------------------------------------===//
  // measuring errors
  //===------------------------------------------------------------------------------------------===//

  // the current test case, or rather, the analytical solution (to the current test case) is only
  // valid if the equilat triangles are aligned with the x/y axis. this is the case for the atlas
  // mesh generator, but not necessarily for imported netcdf meshes. for netcdf meshes the current
  // best bet is to simply compare against a manual computation of the same FD laplacian

  if(!readMeshFromDisk) {
    {
      auto [Linf, L1, L2] = MeasureErrors(wrapper.innerEdges(mesh), nabla2_sol, nabla2, k_size);
      printf("[lap] dx: %e L_inf: %e L_1: %e L_2: %e\n", 180. / nx, Linf, L1, L2);
    }

    if(nx == 340 && ny == 340) {
      auto [Linf, L1, L2] = MeasureErrors("kh_smag_ref.txt", kh_smag, mesh.edges().size(), k_size);
      printf("[kh_smag] dx: %e L_inf: %e L_1: %e L_2: %e\n", 180. / nx, Linf, L1, L2);
    }
  } else {
    std::vector<dawn::float_type> diamondManual(mesh.edges().size());
    for(int level = 0; level < k_size; level++) {
      for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
        auto diamondNbh = atlasInterface::getNeighbors(
            atlasInterface::atlasTag{}, mesh,
            {dawn::LocationType::Edges, dawn::LocationType::Cells, dawn::LocationType::Vertices},
            edgeIdx);

        if(diamondNbh.size() < 4) {
          diamondManual[edgeIdx] = 0.;
        }

        std::vector<dawn::float_type> diamondVals = {
            u(diamondNbh[0], level) + v(diamondNbh[0], level),
            u(diamondNbh[1], level) + v(diamondNbh[1], level),
            u(diamondNbh[2], level) + v(diamondNbh[2], level),
            u(diamondNbh[3], level) + v(diamondNbh[3], level)};
        double hx = 0.5 * wrapper.edgeLength(mesh, edgeIdx);
        double hy = 0.5 * wrapper.vertVertLength(mesh, edgeIdx);

        double fxx = (diamondVals[0] + diamondVals[1] - 2 * (vn(edgeIdx, level))) / (hx * hx);
        double fyy = (diamondVals[2] + diamondVals[3] - 2 * (vn(edgeIdx, level))) / (hy * hy);

        diamondManual[edgeIdx] = fxx + fyy;
      }
    }
    auto [Linf, L1, L2] = MeasureErrors(wrapper.innerEdges(mesh), diamondManual, nabla2, k_size);
    printf("%e %e %e\n", Linf, L1, L2);
  }

  return 0;
}

std::tuple<dawn::float_type, dawn::float_type, dawn::float_type>
MeasureErrors(const std::string& refFname, const atlasInterface::Field<dawn::float_type>& sol,
              int num_edges, int k_size) {
  dawn::float_type Linf = 0.;
  dawn::float_type L1 = 0.;
  dawn::float_type L2 = 0.;

  FILE* fp = fopen(refFname.c_str(), "r");
  std::vector<dawn::float_type> refVec(num_edges);
  for(int edgeIdx = 0; edgeIdx < num_edges; edgeIdx++) {
    double in;
    fscanf(fp, "%lf ", &in);
    refVec[edgeIdx] = in;
  }
  fclose(fp);

  for(int level = 0; level < k_size; level++) {
    for(int edgeIdx = 0; edgeIdx < num_edges; edgeIdx++) {
      dawn::float_type dif = sol(edgeIdx, level) - refVec[edgeIdx];
      Linf = fmax(fabs(dif), Linf);
      L1 += fabs(dif);
      L2 += dif * dif;
    }
  }
  L1 /= (num_edges * k_size);
  L2 = sqrt(L2) / sqrt(num_edges * k_size);
  return {Linf, L1, L2};
}

std::tuple<dawn::float_type, dawn::float_type, dawn::float_type>
MeasureErrors(std::vector<int> indices, const atlasInterface::Field<dawn::float_type>& ref,
              const atlasInterface::Field<dawn::float_type>& sol, int k_size) {
  dawn::float_type Linf = 0.;
  dawn::float_type L1 = 0.;
  dawn::float_type L2 = 0.;
  for(int level = 0; level < k_size; level++) {
    for(int idx : indices) {
      dawn::float_type dif = ref(idx, level) - sol(idx, level);
      Linf = fmax(fabs(dif), Linf);
      L1 += fabs(dif);
      L2 += dif * dif;
    }
  }
  L1 /= (indices.size() * k_size);
  L2 = sqrt(L2) / sqrt(indices.size() * k_size);
  return {Linf, L1, L2};
}

std::tuple<dawn::float_type, dawn::float_type, dawn::float_type>
MeasureErrors(std::vector<int> indices, const std::vector<dawn::float_type>& ref,
              const atlasInterface::Field<dawn::float_type>& sol, int k_size) {
  double Linf = 0.;
  double L1 = 0.;
  double L2 = 0.;
  for(int level = 0; level < k_size; level++) {
    for(int idx : indices) {
      double dif = ref[idx] - sol(idx, level);
      Linf = fmax(fabs(dif), Linf);
      L1 += fabs(dif);
      L2 += dif * dif;
    }
  }
  L1 /= indices.size();
  L2 = sqrt(L2) / sqrt(indices.size());
  return {Linf, L1, L2};
}
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_stencil LANGUAGES CXX CUDA)

find_package(atlas CONFIG REQUIRED)
find_package(atlas_utils CONFIG REQUIRED)
add_library(diamond_stencil STATIC diamond_stencil.cu)
target_link_libraries(diamond_stencil PUBLIC atlas)
target_compile_features(diamond_stencil PUBLIC cxx_std_14)
set_target_properties(diamond_stencil PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(diamond_stencil PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70>)

add_executable(diamond_driver diamond_driver.cpp)
target_link_libraries(diamond_driver PRIVATE diamond_stencil)
target_compile_features(diamond_driver PUBLIC cxx_std_17)
target_link_libraries(diamond_driver PUBLIC atlasUtilsLib)



cmake_minimum_required(VERSION 3.13)
project(sogmm_open3d
	VERSION 0.0.0
	DESCRIPTION
	"SOGMM Open3D"
	LANGUAGES CXX C)

# configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-std=c++17 -march=native -O3 -fopenmp -lstdc++fs")
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# find dependencies
find_package(pybind11 REQUIRED)
find_package(Open3D REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(self_organizing_gmm REQUIRED)
find_package(nanoflann REQUIRED)
find_package(GSL REQUIRED)
find_package(CUDA REQUIRED)

# Setup pybind modules
pybind11_add_module(gmm_open3d_py src/pybind/gmm_open3d_py.cpp)
target_include_directories(gmm_open3d_py
  PRIVATE
  include
  ${sogmm_INCLUDE_DIR}
  ${GSL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(
  gmm_open3d_py
  PRIVATE stdc++fs
  PRIVATE ${GSL_LIBRARIES}
  PRIVATE Eigen3::Eigen
  PRIVATE Open3D::Open3D
)
set_target_properties(gmm_open3d_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

pybind11_add_module(sogmm_gpu src/pybind/sogmm_gpu.cpp)
target_include_directories(sogmm_gpu
  PRIVATE
  include
  ${sogmm_INCLUDE_DIR}
  ${GSL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(
  sogmm_gpu
  PRIVATE stdc++fs
  PRIVATE ${GSL_LIBRARIES}
  PRIVATE Eigen3::Eigen
  PRIVATE Open3D::Open3D
  PRIVATE nanoflann::nanoflann
)
set_target_properties(sogmm_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

# Install
install(TARGETS DESTINATION ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION "include/${PROJECT_NAME}"
  FILES_MATCHING PATTERN "*.h"
)

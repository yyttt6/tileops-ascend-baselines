cmake_minimum_required(VERSION 3.20)
project(sgl_unidex_compat LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)
set(SOC_VERSION Ascend950DT_9592)
set(SGL_ROOT /root/dyq/workspace/ascend_baselines/sgl-kernel-npu)
if(SGL_OFFLINE_CONFIG)
  # These include roots and ABI come from the inspected original build flags.
  # Bypass config_envs.cmake's framework imports for CPU-only compilation.
  if(NOT TORCH_DIR OR NOT TORCH_NPU_DIR OR NOT DEFINED SGL_TORCH_ABI)
    message(FATAL_ERROR "Offline configuration requires explicit framework roots and ABI")
  endif()
  add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=${SGL_TORCH_ABI})
else()
  include(${SGL_ROOT}/cmake/config_envs.cmake)
endif()
include(${SGL_ROOT}/cmake/config_ascend.cmake)
set(ASCEND_INCLUDE_DIR ${ASCEND_HOME_PATH}/include)
ascendc_library(unidex_kernel STATIC
  ${SGL_ROOT}/csrc/sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp)
add_library(sgl_unidex_compat SHARED binding.cpp
  ${SGL_ROOT}/csrc/sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp)
target_include_directories(sgl_unidex_compat PRIVATE
  ${SGL_ROOT}/csrc/utils
  ${ASCEND_INCLUDE_DIR} ${ASCEND_INCLUDE_DIR}/external
  ${ASCEND_INCLUDE_DIR}/experiment/platform ${ASCEND_INCLUDE_DIR}/experiment/runtime
  ${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include ${TORCH_NPU_DIR}/include)
target_compile_options(sgl_unidex_compat PRIVATE
  -O3 -include${SGL_ROOT}/csrc/utils/cann_graph_compat.h)
target_link_directories(sgl_unidex_compat PRIVATE
  ${TORCH_DIR}/lib ${TORCH_NPU_DIR}/lib ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(sgl_unidex_compat PRIVATE unidex_kernel
  torch_npu torch torch_cpu c10 ascendcl tiling_api register platform ascendalog dl)

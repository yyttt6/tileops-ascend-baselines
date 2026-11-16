# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

##############################################################################
# ops-tensor CMake 配置 - Header-only Library
##############################################################################

cmake_minimum_required(VERSION 3.16)

##############################################################################
# 一、基础配置
##############################################################################

# 设置三方库路径
if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party CACHE STRING "CANN third party lib path")
endif()

# 获取 cann-cmake 模块集合
include(cmake/third_party/fetch_cann_cmake.cmake)

# 项目声明
project(ops_tensor VERSION 1.0.0 LANGUAGES CXX)

# 包含变量配置
include(cmake/func.cmake)
include(cmake/variables.cmake)

##############################################################################
# 二、环境初始化
##############################################################################

include(cmake/init_env.cmake)
init_env()  # 发现算子

##############################################################################
# 三、测试构建（可选）
##############################################################################

option(BUILD_TESTING "Build tests" OFF)
option(ENABLE_PACKAGE "Build package" OFF)

if(ENABLE_PACKAGE)
    message(STATUS "Package: ENABLED (SOC: ${SOC_VERSION})")
else()
    message(STATUS "Package: DISABLED")
endif()

if(BUILD_TESTING)
    # Kernel UT（gtest + tikicpulib CPU debug，独立子工程，也可单独 cmake 构建）
    add_subdirectory(tests/ut/op_kernel)
endif()

##############################################################################
# 四、安装规则（Header-only）
##############################################################################

# 安装公共头文件到 ops_tensor/include/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${OPS_TENSOR_INC_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

##############################################################################
# 五、打包配置
##############################################################################

option(ENABLE_PACKAGE "Enable package generation" OFF)
# 打包逻辑在 build.sh 中实现

##############################################################################
# 六、构建摘要
##############################################################################

message(STATUS "")
message(STATUS "==========================================")
message(STATUS "  Build Summary")
message(STATUS "==========================================")
message(STATUS "  Version   : ${PROJECT_VERSION}")
message(STATUS "  Type      : Header-only Library")
message(STATUS "  Operators : ${BUILD_OPERATORS}")
message(STATUS "  Tests     : ${BUILD_TESTING}")
message(STATUS "==========================================")

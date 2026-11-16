# ----------------------------------------------------------------------------------------------------------	
# Copyright (c) 2025 Huawei Technologies Co., Ltd.	
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of	
# CANN Open Software License Agreement Version 2.0 (the "License").	
# Please refer to the License for details. You may not use this file except in compliance with the License.	
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,	
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.	
# See LICENSE in the root of the software repository for the full text of the License.	
# ----------------------------------------------------------------------------------------------------------
project (asc_op_compile_base)

#build asc_op_compile_base whl pkg
message(STATUS "CMAKE_BINARY_DIR=" ${CMAKE_BINARY_DIR})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=" ${CMAKE_CURRENT_SOURCE_DIR})

set(ASC_OP_COMPILE_BASE_WHL_NAME "asc_op_compile_base-0.1.0-py3-none-any.whl")
set(ASC_OP_COMPILE_BASE_DIR "./asc_op_compile_base_python")
set(ASC_OP_COMPILE_BASE_SRC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(asc_platform SHARED ${CMAKE_CURRENT_SOURCE_DIR}/c_api/asc_platform.cc ${CMAKE_CURRENT_SOURCE_DIR}/c_api/asc_platform_api.cpp)
if(BUILD_OPEN_PROJECT)
  target_include_directories(asc_platform PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/base
  )
else()
  target_include_directories(asc_platform PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api
    ${TOP_DIR}/asc/asc-devkit/tools/build/common
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/abl/mmpa/inc
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/abl/slog/inc/toolchain
  )
endif()

target_link_libraries(asc_platform PRIVATE 
  platform
  mmpa
  unified_dlog
  error_manager
)
target_compile_options(asc_platform PRIVATE 
  -fPIC
  -fstack-protector-strong
  -D_FORTIFY_SOURCE=2
  -Os
  -D_GLIBCXX_USE_CXX11_ABI=0
  -Werror
)

target_link_options(asc_platform PRIVATE
  -Wl,-z,relro,-z,now,-z,noexecstack
)

# 将共享库复制到打包 asc_op_compile_base whl包的根目录下
add_custom_command(
  TARGET asc_platform POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Source file: $<TARGET_FILE:asc_platform>"
  COMMAND ${CMAKE_COMMAND} -E echo "Destination file: ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/c_api/libasc_platform.so"
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/c_api
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:asc_platform> ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/c_api/libasc_platform.so
  COMMAND ${CMAKE_STRIP} -s ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/c_api/libasc_platform.so
  COMMAND ${CMAKE_COMMAND} -E echo "Copied libasc_platform.so to ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/c_api/libasc_platform.so"
)

# 将.so复制回c_api，便于测试
add_custom_target(asc_platform_test
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:asc_platform> ${ASC_OP_COMPILE_BASE_SRC}/c_api/libasc_platform.so
  COMMAND ${CMAKE_STRIP} -s ${ASC_OP_COMPILE_BASE_SRC}/c_api/libasc_platform.so
  COMMAND ${CMAKE_COMMAND} -E echo "Copied libasc_platform.so to ${ASC_OP_COMPILE_BASE_SRC}/c_api/libasc_platform.so"
)

# 生成 Python whl pkg
add_custom_target(${ASC_OP_COMPILE_BASE_WHL_NAME} ALL
  COMMAND echo "[ASC_OP_COMPILE_BASE] Build target ${ASC_OP_COMPILE_BASE_WHL_NAME}" && pwd && 
    mkdir -p ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/asc_op_compiler && 
    cp -rf ${ASC_OP_COMPILE_BASE_SRC}/asc_op_compiler/* ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/asc_op_compiler/ &&
    cp -rf ${ASC_OP_COMPILE_BASE_SRC}/c_api/*.py ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/c_api/ &&
    mkdir -p ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/common && 
    cp -rf ${ASC_OP_COMPILE_BASE_SRC}/common/* ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base/common/ &&
    cp ${ASC_OP_COMPILE_BASE_SRC}/__init__.py ${ASC_OP_COMPILE_BASE_DIR}/asc_op_compile_base &&
    cp ${ASC_OP_COMPILE_BASE_SRC}/setup.py ${ASC_OP_COMPILE_BASE_DIR}/ &&
    cd ${ASC_OP_COMPILE_BASE_DIR} && ${HI_PYTHON} setup.py bdist_wheel && cd - &&
    ls && pwd &&
    mkdir ${CMAKE_INSTALL_PREFIX}/lib -p &&
    ${CMAKE_COMMAND} -E rename ${ASC_OP_COMPILE_BASE_DIR}/dist/asc_op_compile_base*.whl ${CMAKE_INSTALL_PREFIX}/lib/${ASC_OP_COMPILE_BASE_WHL_NAME} &&
    echo "[ASC_OP_COMPILE_BASE] Build target ${ASC_OP_COMPILE_BASE_WHL_NAME} end"
  DEPENDS asc_platform
)

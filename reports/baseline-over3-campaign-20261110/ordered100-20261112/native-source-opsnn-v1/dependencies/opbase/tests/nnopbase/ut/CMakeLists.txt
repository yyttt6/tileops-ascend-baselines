# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
project(nnopbase_ut)
cmake_minimum_required(VERSION 3.14.1)
set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(__CCE_KT_TEST__=1)
add_compile_definitions(NNOPBASE_UT)

message(STATUS "nnopbase_ut_test compile start")

# Ascend mode
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    if ("$ENV{USER}" STREQUAL "root")
      set(ASCEND_PATH /usr/local/Ascend/cann)
    else ()
      set(ASCEND_PATH $ENV{HOME}/Ascend/cann)
    endif ()
endif()
message(STATUS "Search libs under install path ${ASCEND_PATH}")

# include path
set(ASCEND_INCLUDE_DIR ${ASCEND_PATH}/include)
set(ASCEND_PKG_INC_DIR ${ASCEND_PATH}/pkg_inc)

# local path
set(NNOPBASE_UT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OPS_BASE_PATH ${NNOPBASE_UT_DIR}/../../..)
get_filename_component(OPS_BASE_PATH "${OPS_BASE_PATH}" ABSOLUTE)
message(STATUS "NNOPBASE_UT_DIR: ${NNOPBASE_UT_DIR}")
message(STATUS "OPS_BASE_PATH: ${OPS_BASE_PATH}")

# generate protobuff files
file(GLOB utils_proto_src_files ${OPS_BASE_PATH}/src/nnopbase/aicpu/proto/*.proto)
protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${utils_proto_src_files})
file(GLOB cpu_proto_src_files ${OPS_BASE_PATH}/src/nnopbase/aicpu/cpu_proto/*.proto)
protobuf_generate(aicpu_proto CPU_PROTO_SRCS CPU_PROTO_HDRS ${cpu_proto_src_files})
message(STATUS "generate aicpu successfully")

# op_proto ut
file(GLOB op_proto_so_ut_stub_files ${NNOPBASE_UT_DIR}/../common/depends/op_proto_stub/op_proto_stub.cpp)
add_library(op_proto_so_ut_stub_rt2.0 SHARED ${op_proto_so_ut_stub_files})
if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set(PROTO_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../composite_op/built-in/op_proto/lib/linux/x86_64)
else()
  set(PROTO_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../composite_op/built-in/op_proto/lib/linux/aarch64)
endif()
target_compile_options(op_proto_so_ut_stub_rt2.0 PRIVATE)
add_custom_command(
    TARGET op_proto_so_ut_stub_rt2.0 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:op_proto_so_ut_stub_rt2.0> ${PROTO_STUB_DIR}/libop_proto_so_ut_stub_rt2.0.so
)

# op_tiling ut
file(GLOB op_tiling_ut_so_stub_files ${NNOPBASE_UT_DIR}/../common/depends/op_tiling_stub/op_tiling_stub.cpp)
add_library(op_tiling_ut_so_stub_rt2.0 SHARED ${op_tiling_ut_so_stub_files})
if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
  set(TILING_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../composite_op/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64)
else()
  set(TILING_STUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../composite_op/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64)
endif()
target_compile_options(op_tiling_ut_so_stub_rt2.0 PRIVATE)
add_custom_command(
    TARGET op_tiling_ut_so_stub_rt2.0 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:op_tiling_ut_so_stub_rt2.0> ${TILING_STUB_DIR}/libop_tiling_ut_so_stub_rt2.0.so
)

set(NNOPBASE_TEST_HEADER_FILES
    ${OPS_BASE_PATH}/src/nnopbase/common/inc/
    ${OPS_BASE_PATH}/src/nnopbase/common/op_info_record/
    ${OPS_BASE_PATH}/src/nnopbase/composite_op/
    ${OPS_BASE_PATH}/src/nnopbase/composite_op/aclnn_engine/
    ${OPS_BASE_PATH}/src/nnopbase/individual_op/
    ${OPS_BASE_PATH}/src/nnopbase/individual_op/api
    ${OPS_BASE_PATH}/src/nnopbase/individual_op/utils
    ${OPS_BASE_PATH}/src/nnopbase/aicpu/task_handler/
    ${OPS_BASE_PATH}/include/
    ${OPS_BASE_PATH}/include/nnopbase/
    ${OPS_BASE_PATH}/include/nnopbase/opdev/
    ${OPS_BASE_PATH}/tests/nnopbase/common/

    ${CANN_3RD_LIB_PATH}/gtest/include/
    ${CANN_3RD_LIB_PATH}/json/include/
    ${CANN_3RD_LIB_PATH}/mockcpp/include/
    ${CANN_3RD_LIB_PATH}/boost/
    ${CMAKE_BINARY_DIR}/mockcpp/include/

    ${ASCEND_INCLUDE_DIR}/
    ${ASCEND_INCLUDE_DIR}/transformer/

    ${ASCEND_PKG_INC_DIR}/
    ${ASCEND_PKG_INC_DIR}/aicpu/
    ${ASCEND_PKG_INC_DIR}/dump/
    ${ASCEND_PKG_INC_DIR}/profiling/
    ${ASCEND_PKG_INC_DIR}/runtime/
    ${ASCEND_PKG_INC_DIR}/runtime/runtime/

    ${CMAKE_BINARY_DIR}/proto/aicpu/
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto/
)

# 测试接口源文件
file(GLOB_RECURSE NNOPBASE_SRC_FILES CONFIGURE_DEPENDS
    ${OPS_BASE_PATH}/src/nnopbase/composite_op/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/common/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/stub/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/tls_guardian/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/aicpu/task_handler/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/op_common/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/individual_op/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/individual_op/api/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/individual_op/utils/*.cpp
    ${OPS_BASE_PATH}/src/nnopbase/individual_op/executor/*.cpp
)

# testcase源文件
file(GLOB_RECURSE NNOPBASE_TEST_COMMON_CASE_SRC_FILES CONFIGURE_DEPENDS
    ${NNOPBASE_UT_DIR}/individual_op/*.cpp                                     # individual_op
    ${NNOPBASE_UT_DIR}/composite_op/*.cpp                                      # composite_op
    # ${NNOPBASE_UT_DIR}/aicpu/*.cpp                                           # aicpu_op
)

# testcase源文件依赖的功能文件
file(GLOB_RECURSE NNOPBASE_TEST_COMMON_SRC_FILES CONFIGURE_DEPENDS
    # ${NNOPBASE_UT_DIR}/../common/depends/*cpp
    ${NNOPBASE_UT_DIR}/../common/utils/*cpp
    ${NNOPBASE_UT_DIR}/../common/depends/metadef/space_registry_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/op/aclnn_bninference_d_kernel_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/op/aclnn_custom_op_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/op/op_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/runtime/runtime_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/dump/dump_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/platform/platform_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/acl/aclrt_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/profiler/profiler_stub.cpp
    ${NNOPBASE_UT_DIR}/../common/depends/mmpa/mmpa_stub.cpp
)

# 创建utest执行对象
add_executable(nnopbase_utest
    ${NNOPBASE_UT_DIR}/main.cpp
    ${NNOPBASE_SRC_FILES}
    ${NNOPBASE_TEST_COMMON_CASE_SRC_FILES}
    ${NNOPBASE_TEST_COMMON_SRC_FILES}
    ${PROTO_HDRS}
    ${CPU_PROTO_HDRS}
)

target_compile_definitions(nnopbase_utest PRIVATE RUN_TEST TLS_GUARDIAN_ __LLT__)

target_include_directories(nnopbase_utest BEFORE PRIVATE
    ${NNOPBASE_TEST_HEADER_FILES}
)

target_compile_options(nnopbase_utest PRIVATE
    -g
    -fno-omit-frame-pointer
    -fno-stack-protector
    -fno-sanitize-recover
    -fno-stack-check
    # -fsanitize=undefined
    -fno-sanitize=vptr,alignment
    -fno-access-control
)

target_link_libraries(nnopbase_utest PRIVATE
    intf_pub
    protobuf_static
    -Wl,--no-as-needed
    op_tiling_ut_so_stub_rt2.0
    op_proto_so_ut_stub_rt2.0
    ${ASCEND_HOME_PATH}/lib64/libruntime.so
    ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    ${ASCEND_HOME_PATH}/lib64/libascend_dump.so
    ${ASCEND_HOME_PATH}/lib64/libc_sec.so
    ${ASCEND_HOME_PATH}/lib64/libexe_graph.so
    ${ASCEND_HOME_PATH}/lib64/libacl_rt.so
    ${ASCEND_HOME_PATH}/lib64/libmetadef.so
    ${ASCEND_HOME_PATH}/lib64/liberror_manager.so
    ${ASCEND_HOME_PATH}/lib64/libopp_registry.so
    ${ASCEND_HOME_PATH}/lib64/libmmpa.so
    ${ASCEND_HOME_PATH}/lib64/libprofapi.so
    ${ASCEND_HOME_PATH}/lib64/libplatform.so
    ${ASCEND_HOME_PATH}/lib64/libregister.so
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libascend_protobuf.so
    -Wl,--as-needed
    json
    gtest
    gtest_main
    -ldl
    $<$<BOOL:${ENABLE_COVERAGE}>:gcov>
    ${opInfoRecordLinkLibraries}
    mockcpp
    ascend_protobuf_static
 PUBLIC
)

target_compile_definitions(nnopbase_utest PRIVATE google=ascend_private)

set(NNOPBASE_LLT_MOCK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mock)
get_filename_component(NNOPBASE_LLT_MOCK_DIR "${NNOPBASE_LLT_MOCK_DIR}" ABSOLUTE)
set(NNOPBASE_LLT_DEPENDS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common/depends)
get_filename_component(NNOPBASE_LLT_DEPENDS_DIR "${NNOPBASE_LLT_DEPENDS_DIR}" ABSOLUTE)
target_compile_definitions(nnopbase_utest
  PRIVATE
    OP_API_COMMON_UT_SRC_DIR=\"${NNOPBASE_LLT_MOCK_DIR}\"
    NNOPBASE_UT_DEPENDS_DIR=\"${NNOPBASE_LLT_DEPENDS_DIR}\"
)

if(ENABLE_COVERAGE)
    target_compile_options(nnopbase_utest PRIVATE
    --coverage
    -fprofile-arcs
    -ftest-coverage)
endif()
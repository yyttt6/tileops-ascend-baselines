# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

project(nnopbase C CXX)

# local path
set(NNOPBASE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(OPS_BASE_PATH ${NNOPBASE_PATH}/../..)
get_filename_component(OPS_BASE_PATH "${OPS_BASE_PATH}" ABSOLUTE)

add_library(nnopbase_headers INTERFACE)
target_include_directories(nnopbase_headers INTERFACE
    ${OPS_BASE_PATH}/include/nnopbase/
    ${OPS_BASE_PATH}/include/nnopbase/opdev
)

# local headers target
add_library(nnopbase_internal_headers INTERFACE)
target_include_directories(nnopbase_internal_headers INTERFACE
    ${NNOPBASE_PATH}/individual_op/
    ${NNOPBASE_PATH}/composite_op/
    ${NNOPBASE_PATH}/common/inc
)

# nnopbase
add_library(nnopbase SHARED)

# source code compile in yellow zone
target_include_directories(nnopbase BEFORE  PRIVATE 
    ${TOP_DIR}/runtime/pkg_inc/
    ${TOP_DIR}/runtime/pkg_inc/runtime/
    ${TOP_DIR}/runtime/pkg_inc/aicpu_sched/common/
    ${TOP_DIR}/runtime/include/external/
    ${TOP_DIR}/runtime/src/inc/runtime/
    ${TOP_DIR}/metadef/inc/external/
    ${TOP_DIR}/metadef/inc/
    ${TOP_DIR}/hcomm/pkg_inc/
    ${TOP_DIR}/hcomm/pkg_inc/legacy/
    ${TOP_DIR}/hcomm/include/
    ${TOP_DIR}/air/graph_metadef/third_party/transformer/inc/
    ${TOP_DIR}/abl/libc_sec/src/
)

target_link_libraries(nnopbase
    PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:cann_version_headers>
        nnopbase_internal_headers
        -Wl,--no-as-needed
        runtime
        unified_dlog
        ascend_dump
        c_sec
        exe_graph
        acl_rt
        metadef
        error_manager
        opp_registry
        graph
        register
        dummy_tls
        -Wl,--as-needed
        json
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        ${opInfoRecordLinkLibraries}
        ascend_protobuf_shared_headers
    PUBLIC
        $<BUILD_INTERFACE:nnopbase_headers>
)

target_compile_definitions(nnopbase PRIVATE google=ascend_private ACLNN_LOG_FMT_CHECK LOG_CPP)
target_compile_options(nnopbase PRIVATE
    -O2 -fno-common -Wextra -Wfloat-equal -D_FORTIFY_SOURCE=2)

# common
add_subdirectory(common)

# tls_guardian
add_subdirectory(tls_guardian)

# composite_op
add_subdirectory(composite_op)

# individual_op
add_subdirectory(individual_op)

# because aicpu proto call add_custom_command for nnopbase so it must be placed here
# generate protobuff files
set(AICPU_PATH ${NNOPBASE_PATH}/aicpu)
file(GLOB utils_proto_src_files ${AICPU_PATH}/proto/*.proto)
protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${utils_proto_src_files})
file(GLOB cpu_proto_src_files ${AICPU_PATH}/cpu_proto/*.proto)
protobuf_generate(aicpu_proto CPU_PROTO_SRCS CPU_PROTO_HDRS ${cpu_proto_src_files})
# aicpu
add_subdirectory(aicpu)

# stub
add_subdirectory(stub)

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(STUB_SRC_LIST
    ${CMAKE_CURRENT_BINARY_DIR}/stub_aicpu_task.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_common_types.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_data_type_utils.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_format_utils.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_framework_op.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_op_def.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_op_dfx.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_op_executor.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_platform.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_shape_utils.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_tensor_view_utils.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_individual_op_api.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_acl_meta.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_bridge_pool.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_block_pool.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_aicpu_args_handler.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_fp16_t.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_op_log.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_block_store.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_op_cache.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_op_arg_def.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_pool_allocator.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_object.cc
)

set(HI_PYTHON python3)

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "HI_PYTHON: ${HI_PYTHON}")
message(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")

add_custom_command(
    OUTPUT ${STUB_SRC_LIST}
    COMMAND echo "Generating nnopbase stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/gen_stubapi.py ${OPS_BASE_PATH} ${CMAKE_CURRENT_BINARY_DIR} > /dev/null 2>&1
            && echo "Generating nnopbase stub files end."      
)


add_custom_target(nnopbase_stub DEPENDS ${STUB_SRC_LIST})

add_library(stub_internal_headers INTERFACE)
target_include_directories(stub_internal_headers INTERFACE
    ${OPS_BASE_PATH}/include/nnopbase/aclnn
    ${OPS_BASE_PATH}/include/nnopbase/opdev/aicpu
    ${NNOPBASE_PATH}/individual_op/api
    ${NNOPBASE_PATH}/composite_op/aclnn_engine
    ${NNOPBASE_PATH}/composite_op/mem_mgr/
    ${NNOPBASE_PATH}/common
)

add_library(stub_nnopbase SHARED ${STUB_SRC_LIST})

add_dependencies(stub_nnopbase nnopbase_stub)

# source code compile in yellow zone
target_include_directories(stub_nnopbase PRIVATE
    ${TOP_DIR}/runtime/pkg_inc/
    ${TOP_DIR}/runtime/pkg_inc/runtime/
    ${TOP_DIR}/runtime/pkg_inc/aicpu_sched/common/
    ${TOP_DIR}/runtime/include/external/
    ${TOP_DIR}/runtime/include/dfx
    ${TOP_DIR}/runtime/src/inc/runtime/
    ${TOP_DIR}/metadef/inc/external/
    ${TOP_DIR}/metadef/inc/
    ${TOP_DIR}/hcomm/pkg_inc/
    ${TOP_DIR}/hcomm/pkg_inc/legacy/
    ${TOP_DIR}/hcomm/include/
    ${TOP_DIR}/air/graph_metadef/third_party/transformer/inc/
    ${TOP_DIR}/abl/libc_sec/src/
)

target_compile_options(stub_nnopbase PRIVATE -O2 -fno-common -Wno-extra -Wfloat-equal -Wno-unused-parameter -D_FORTIFY_SOURCE=2)

target_link_libraries(stub_nnopbase
    PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:error_manager_headers>
        metadef_headers
        runtime_headers
        nnopbase_internal_headers
        stub_internal_headers
        c_sec
        json
    PUBLIC
        $<BUILD_INTERFACE:nnopbase_headers>
)

set_target_properties(stub_nnopbase PROPERTIES
    OUTPUT_NAME nnopbase
    LIBRARY_OUTPUT_DIRECTORY stub
)

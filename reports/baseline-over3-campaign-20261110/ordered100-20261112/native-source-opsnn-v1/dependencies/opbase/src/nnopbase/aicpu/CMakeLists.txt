# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB aicpu_files ${AICPU_PATH}/task_handler/*.cpp)

# set proto include path
set(AICPU_PROTO_INCLUDING_PATH
    ${CMAKE_BINARY_DIR}/proto/aicpu/
    ${CMAKE_BINARY_DIR}/proto/aicpu_proto/
    ${NNOPBASE_PATH}/../include/
    ${NNOPBASE_PATH}/../include/nnopbase/
    ${NNOPBASE_PATH}/composite_op/
    ${NNOPBASE_PATH}/common/
    ${ASCEND_HOME_PATH}/include/experiment/datagw
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/aicpu
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${TOP_DIR}/runtime/pkg_inc/aicpu_sched/common
)

target_sources(nnopbase PRIVATE ${aicpu_files} ${PROTO_HDRS} ${CPU_PROTO_HDRS})
target_include_directories(nnopbase PRIVATE ${AICPU_PROTO_INCLUDING_PATH})
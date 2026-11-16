# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# find all source files

set(COMMON_API_PATH ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE NNOPBASE_COMMMON_SRCS CONFIGURE_DEPENDS "${COMMON_API_PATH}/*.cpp")


MESSAGE(STATUS "COMMON_API_PATH: ${COMMON_API_PATH}")

# op_info_record
add_subdirectory(op_info_record)

# set include path
set(NNOPBASE_COMMON_INCLUDING_PATH
    ${COMMON_API_PATH}/
    ${COMMON_API_PATH}/api/
    ${COMMON_API_PATH}/inc/

    ${ASCEND_INCLUDE_DIR}/adump
)

target_sources(nnopbase PRIVATE ${NNOPBASE_COMMMON_SRCS})
target_include_directories(nnopbase PUBLIC ${NNOPBASE_COMMON_INCLUDING_PATH})
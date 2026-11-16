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
file(GLOB_RECURSE NNOPBASE_INDIVIDUAL_OP_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
# set include path
set(NNOPBASE_INDIVIDUAL_OP_INCLUDING_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/inc
    ${ASCEND_INCLUDE_DIR}/acl
    ${ASCEND_INCLUDE_DIR}/external/ac
)

if(DEFINED PRODUCT_SIDE AND "${PRODUCT_SIDE}" STREQUAL "device")
    target_compile_definitions(nnopbase PRIVATE PRODUCT_SIDE_IS_DEVICE=1)
endif()
target_sources(nnopbase PRIVATE ${NNOPBASE_INDIVIDUAL_OP_SRCS})
target_include_directories(nnopbase PRIVATE ${NNOPBASE_INDIVIDUAL_OP_INCLUDING_PATH})
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.14.1)

# Ascend mode
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()
message(STATUS "Search libs under install path ${ASCEND_HOME_PATH}")

# root path
set(OPS_BASE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)
get_filename_component(OPS_BASE_PATH "${OPS_BASE_PATH}" ABSOLUTE)
message(STATUS "OPS_BASE_PATH: ${OPS_BASE_PATH}")


# path for downloading third party
if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${OPS_BASE_PATH}/third_party)
endif()

set(FORCE_REBUILD_CANN_3RD TRUE)
message(STATUS "CANN_3RD_LIB_PATH: ${CANN_3RD_LIB_PATH}")

# path of op common headers
if(NOT OPS_BASE_INCLUDE)
    set(OPS_BASE_INCLUDE ${OPS_BASE_PATH}/pkg_inc)
endif()

add_cann_third_party(json)

add_subdirectory(op_common)
add_subdirectory(nnopbase)

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16.0)
project(host_bisheng)

include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_intf.cmake)

include(${BUILD_CFG})

add_library(host_bisheng_obj OBJECT
    ${SOURCES}
)

target_compile_options(host_bisheng_obj PRIVATE
    "SHELL:-include ${INCLUDE_DIR}/aclrtlaunch_triple_chevrons_func.h"
)

target_link_libraries(host_bisheng_obj PRIVATE
    $<BUILD_INTERFACE:device_intf_pub>
    $<BUILD_INTERFACE:intf_host_bisheng>
)

install(TARGETS host_bisheng_obj
    DESTINATION .
)

# Public compile rules of device
add_library(intf_host_bisheng INTERFACE)

if(BUILD_MODE STREQUAL "c310")
    target_compile_options(intf_host_bisheng INTERFACE
        ${ASCENDC_HOST_COMPILE_OPTIONS}
        -fPIC
        --cce-host-only
        -D__ASC_NPU_HOST__
        -fcce-kernel-launch-custom
        ${OPTIONS}
    )
else()
    target_compile_options(intf_host_bisheng INTERFACE
        ${ASCENDC_HOST_COMPILE_OPTIONS}
        -fPIC
        --cce-host-only
        -fcce-kernel-launch-custom
        ${OPTIONS}
    )
endif()

target_include_directories(intf_host_bisheng INTERFACE 
    ${INCLUDES}
)

target_compile_definitions(intf_host_bisheng INTERFACE
    ${DEFINITIONS}
)


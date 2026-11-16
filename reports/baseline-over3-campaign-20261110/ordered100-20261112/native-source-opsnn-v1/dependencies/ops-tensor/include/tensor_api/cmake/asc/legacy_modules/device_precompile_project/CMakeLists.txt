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
project(device_precompile)

include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_intf.cmake)

if(NOT "${TARGET_TYPE}" STREQUAL "OBJECT")
    set(DEFINITION_OPTION --generate-definition)
else()
    set(DEFINITION_OPTION --generate-ktype-section)
endif()

# Generate device-side precompiled files
add_library(precompile_obj
    OBJECT ${SOURCES}
)

# Add the -E option based on normal compilation options to generate precompiled files
target_compile_options(precompile_obj PRIVATE
    -E -includestdio.h
)

target_compile_definitions(precompile_obj PRIVATE
    __CHECK_FEATURE_AT_PRECOMPILE
)

if(${BUILD_MODE} STREQUAL "c310" OR ${BUILD_MODE} STREQUAL "c220")
    target_link_libraries(precompile_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_aiv_intf_pub>
        $<BUILD_INTERFACE:intf_device>
    )
else()
    target_link_libraries(precompile_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_intf_pub>
        $<BUILD_INTERFACE:intf_device>
    )
endif()

string(REPLACE "," " " precompile_objs "$<TARGET_OBJECTS:precompile_obj>")

# Generate template_config.cmake, set source file properties with special compile options
add_custom_target(check_src_template ALL
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} extract_src_template.py ${precompile_objs} -d ${DST_DIR}
    WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
    DEPENDS precompile_obj
    COMMAND_EXPAND_LISTS
)

# Precompiling uses default optimization options, Exclude user-defined optimization options
set(OPTIMIZATION_OPTIONS -O0 -O1 -O2 -O3 -Os --cce-ignore-always-inline=true)
set(COMPILE_OPTIONS)
foreach (_option ${OPTIONS})
    if(NOT "${_option}" IN_LIST OPTIMIZATION_OPTIONS)
        list(APPEND COMPILE_OPTIONS ${_option})
    endif()
endforeach()

# Public compile rules of device
add_library(intf_device INTERFACE)

target_compile_options(intf_device INTERFACE
    ${COMPILE_OPTIONS}
)

target_include_directories(intf_device INTERFACE 
    ${INCLUDES}
)

target_compile_definitions(intf_device INTERFACE
    ${DEFINITIONS}
)


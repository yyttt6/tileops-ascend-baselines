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
project(device_preprocessed)

option(ENABLE_MERGE "Enable merge objs" OFF)

if(ENABLE_MERGE)
    set(MERGE_OPTION "-m")
endif()

include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_config.cmake)
include(${ASCENDC_KERNEL_CMAKE_DIR}/bisheng_intf.cmake)

include(${BUILD_CFG})

# Flag file, used to determine whether the mode is executed
set(MIX_BUILD_FLAG ${DST_DIR}/mix_build.flag)
set(AIC_BUILD_FLAG ${DST_DIR}/aic_build.flag)
set(AIV_BUILD_FLAG ${DST_DIR}/aiv_build.flag)

add_library(intf_device_obj INTERFACE)

target_compile_options(intf_device_obj INTERFACE
    ${OPTIONS}
)

target_include_directories(intf_device_obj INTERFACE 
    ${INCLUDES}
)

target_compile_definitions(intf_device_obj INTERFACE
    ${DEFINITIONS}
)

file(REMOVE ${MIX_BUILD_FLAG})
file(REMOVE ${AIC_BUILD_FLAG})
file(REMOVE ${AIV_BUILD_FLAG})
file(MAKE_DIRECTORY ${DST_DIR})

if(MIX_SOURCES)
    add_library(device_obj OBJECT
        ${MIX_SOURCES}
    )
    target_compile_options(device_obj PRIVATE
        --cce-disable-kernel-global-attr-check
    )
    # do not modify link order, custom options should be at the end
    target_link_libraries(device_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_intf_pub>
        intf_device_obj
    )

    string(REPLACE "," " " device_objs "$<TARGET_OBJECTS:device_obj>")

    add_custom_target(merge_device_obj ALL
        COMMAND bash merge_obj.sh -l ${CMAKE_LINKER} -o ${DST_DIR} -n device.o ${MERGE_OPTION} -f ${MIX_BUILD_FLAG} ${device_objs}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        DEPENDS device_obj
        COMMAND_EXPAND_LISTS
    )
endif()

if(AIC_SOURCES)
    add_library(device_aic_obj OBJECT
        ${AIC_SOURCES}
    )
    target_compile_options(device_aic_obj PRIVATE
        --cce-disable-kernel-global-attr-check
    )
    # do not modify link order, custom options should be at the end
    target_link_libraries(device_aic_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_intf_pub>
        intf_device_obj
    )

    string(REPLACE "," " " device_aic_objs "$<TARGET_OBJECTS:device_aic_obj>")

    add_custom_target(merge_aic_device_obj ALL
        COMMAND bash merge_obj.sh -l ${CMAKE_LINKER} -o ${DST_DIR} -n device_aic.o ${MERGE_OPTION} -f ${AIC_BUILD_FLAG} ${device_aic_objs}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        DEPENDS device_aic_obj
        COMMAND_EXPAND_LISTS
    )
endif()

if(AIV_SOURCES)
    add_library(device_aiv_obj OBJECT
        ${AIV_SOURCES}
    )
    target_compile_options(device_aiv_obj PRIVATE
        --cce-disable-kernel-global-attr-check
    )
    # do not modify link order, custom options should be at the end
    target_link_libraries(device_aiv_obj PRIVATE
        $<BUILD_INTERFACE:${BUILD_MODE}_intf_pub>
        intf_device_obj
    )

    string(REPLACE "," " " device_aiv_objs "$<TARGET_OBJECTS:device_aiv_obj>")

    add_custom_target(merge_aiv_device_obj ALL
        COMMAND bash merge_obj.sh -l ${CMAKE_LINKER} -o ${DST_DIR} -n device_aiv.o ${MERGE_OPTION} -f ${AIV_BUILD_FLAG} ${device_aiv_objs}
        WORKING_DIRECTORY ${ASCENDC_KERNEL_CMAKE_DIR}/util
        DEPENDS device_aiv_obj
        COMMAND_EXPAND_LISTS
    )
endif()






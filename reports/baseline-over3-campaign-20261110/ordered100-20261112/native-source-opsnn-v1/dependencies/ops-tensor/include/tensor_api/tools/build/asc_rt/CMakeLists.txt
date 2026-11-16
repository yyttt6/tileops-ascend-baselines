# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
add_library(ascend_runtime_base OBJECT ascendc_runtime.cpp aicpu_rt.cpp)

if(BUILD_OPEN_PROJECT)
  target_include_directories(ascend_runtime_base PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include/acl
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
  )
else()
  target_include_directories(ascend_runtime_base PRIVATE
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
    ${TOP_DIR}/ace/npuruntime/inc/
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/metadef/inc/external/
    ${TOP_DIR}/abl/msprof/inc
    ${TOP_DIR}/abl/libc_sec/include
    ${TOP_DIR}/abl/mmpa/inc
    ${TOP_DIR}/abl/slog/inc/toolchain
    ${TOP_DIR}/asc/asc-devkit/tools/build/common
  )
endif()

target_compile_options(ascend_runtime_base PRIVATE
  -Wextra
  -Wfloat-equal
  -fvisibility-inlines-hidden
  -fvisibility=hidden
  -Werror
)

target_link_libraries(ascend_runtime_base PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    mmpa
    unified_dlog
    $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:acl_rt>
    $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:ascendcl>
)

add_library(ascendc_runtime STATIC
   $<TARGET_OBJECTS:ascend_runtime_base>
   $<TARGET_OBJECTS:elf_tool_obj>
)

install(TARGETS ascendc_runtime
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)

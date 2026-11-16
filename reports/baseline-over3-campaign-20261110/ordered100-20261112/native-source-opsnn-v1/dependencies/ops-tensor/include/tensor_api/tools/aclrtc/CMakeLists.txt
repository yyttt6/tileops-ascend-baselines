# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC)

add_library(acl_rtc SHARED
    ${SRC}
)

if(BUILD_OPEN_PROJECT)
    target_include_directories(acl_rtc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl
        ${ASCEND_CANN_PACKAGE_PATH}/include/pkg_inc/base/
        ${CCEC_PATH}/../include
    )
else()
    target_include_directories(acl_rtc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl/
        ${CCEC_PATH}/../include
    )
endif()

target_link_directories(acl_rtc PRIVATE
    ${CCEC_PATH}/../lib
)

target_link_libraries(acl_rtc PRIVATE
    $<BUILD_INTERFACE:intf_pub_cxx17>
    dl
    c_sec_static
)

target_compile_options(acl_rtc PRIVATE
    -fPIE
    -fno-strict-aliasing
    -fno-common
    -Wall
    -Wextra
    -Wfloat-equal
    -Werror
    -Wformat
    -D_FORTIFY_SOURCE=2
    -Os
)

target_link_options(acl_rtc PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -Wl,-Bsymbolic
    -Wl,--gc-sections
    $<$<CONFIG:Release>:-s>
)

install(TARGETS acl_rtc
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/acl_rt_compile.h
    DESTINATION ${INSTALL_LIBRARY_DIR}/acl
)
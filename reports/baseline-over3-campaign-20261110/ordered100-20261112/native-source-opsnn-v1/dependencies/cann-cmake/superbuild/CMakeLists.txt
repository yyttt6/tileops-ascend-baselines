# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.3)
project(cann_superbuild)

# superbuild 入口：负责 host 侧直接 add_subdirectory 编译，并在需要时通过
# ExternalProject 启动独立 CMake 进程交叉编译 device 侧（见文件末尾）。
# 整体流程：解析依赖 -> host 子目录编译 -> (可选) device 交叉编译 -> 打包。
include(../function/function.cmake)
init_cann_superbuild_project(PRODUCT_SIDE "host" PREPEND_MODULE_PATH)
set(BUILD_OPEN_PROJECT TRUE)
set(PACKAGE_OPEN_PROJECT TRUE)
# opbase编译依赖BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG设置为TRUE
set(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG TRUE)
# 默认开启打包
set(ENABLE_PACKAGE TRUE)

# GE编译依赖这个路径，路径区别于子包编译
set(METADEF_INC_PATH "${CANN_TOP_DIR}/metadef/inc/external")
set(BUILD_METADEF TRUE)
# pypto打开pypto_impl在线编译特性
set(ENABLE_FEATURE_PYBIND11_IMPL_COMPILE_ONLINE TRUE)

# 防止package时触发all编译，部分make的profiling工具在重编时会覆盖原有的性能数据
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if(NOT "asc-devkit" IN_LIST CANN_BINARY_PACKAGES AND NOT "all" IN_LIST CANN_BINARY_PACKAGES)
    # 从源码导入asc包
    set(CUSTOM_ASC_CMAKE_PATH "${CANN_TOP_DIR}/${CANN_PACKAGE_DIR_asc-devkit}/cmake/asc")
endif()

# ---------- 依赖解析与 device 编译判定 ----------
# 递归解析 --pkgs 指定包的构建依赖链，结果写入 CANN_DEPEND_PACKAGES；
# calc_device_packages 进一步从中筛出含 cmake/device 子目录的包。
get_pkg_dependencies("${CANN_PACKAGES}")
calc_device_packages()

# ENABLE_BUILD_DEVICE 有两层语义：
# 1. 命令行传入值（build.sh --build_host_only 控制）：表示用户是否要求编译 device 侧
# 2. 子包读取值：表示子包是否应自行编译 device 侧代码
#
# CANN_SUPERBUILD_BUILD_DEVICE 保存命令行原始值供 superbuild 判定是否启动
# device 交叉编译。用户直接调用 cmake（未经 build.sh）时该变量未定义，
# 默认按启用处理（与 build.sh 默认行为一致）。
if(DEFINED ENABLE_BUILD_DEVICE)
    set(CANN_SUPERBUILD_BUILD_DEVICE "${ENABLE_BUILD_DEVICE}")
else()
    set(CANN_SUPERBUILD_BUILD_DEVICE TRUE)
endif()

# 将 ENABLE_BUILD_DEVICE 置 FALSE，防止子包各自启动 device 编译
# （device 编译由 superbuild 的 ExternalProject_Add 统一处理）。
set(ENABLE_BUILD_DEVICE FALSE)

# 仅当命令行允许(--build_host_only 未指定)、且存在需要编译 device 侧的包时，
# 才启用 device 交叉编译。ge-executor/ge-compiler 等纯 host 侧包已在
# calc_device_packages 中排除，DEVICE_CANN_PACKAGES 非空即代表存在真实 device 包。
if(CANN_SUPERBUILD_BUILD_DEVICE AND DEVICE_CANN_PACKAGES)
    set(SUPERBUILD_ENABLE_DEVICE TRUE)
else()
    set(SUPERBUILD_ENABLE_DEVICE FALSE)
endif()

# ---------- host 侧编译 ----------
# 将所有依赖包以 add_subdirectory 引入，直接在当前进程编译（区别于 device 的独立进程）。
get_package_dirs(CANN_DEPEND_PACKAGE_DIRS "${CANN_DEPEND_PACKAGES}")
foreach(CANN_DEPEND_PACKAGE_DIR IN LISTS CANN_DEPEND_PACKAGE_DIRS)
    # 这里不可以使用EXCLUDE_FROM_ALL，cmake会排除install命令，这不是期望的
    add_subdirectory(${CANN_TOP_DIR}/${CANN_DEPEND_PACKAGE_DIR} ${CANN_DEPEND_PACKAGE_DIR})
endforeach()

# set_cann_all_targets 的作用：先将 CMAKE_SOURCE_DIR 下所有 target 标记为
# EXCLUDE_FROM_ALL（默认不参与编译，避免 --pkgs 未指定的冗余包被构建），
# 再为实际要构建的目标建立聚合目标 cann_all_targets 并 add_dependencies，
# 使仅这些目标随 ALL 一起编译。这样未在 --pkgs 中指定的组件会被排除。
get_package_dirs(CANN_PACKAGE_DIRS "${CANN_PACKAGES}")
set_cann_all_targets("${CANN_PACKAGE_DIRS}")

# ---------- device 交叉编译准备 ----------
# ExternalProject 通过命令行传递 CMAKE_ARGS，CMake 列表的分号 ";" 会被破坏，
# 故先将其替换为自定义分隔符 "::"，再由下方 LIST_SEPARATOR :: 还原。
if(NOT DEFINED DEVICE_TOOLCHAIN_DIR)
    set(DEVICE_TOOLCHAIN_DIR ${ASCEND_INSTALL_PATH}/toolkit/toolchain/hcc)
endif()
if(NOT DEFINED DEVICE_TOOLCHAIN_FILE)
    set(DEVICE_TOOLCHAIN_FILE "${CANN_CMAKE_DIR}/toolchain/aarch64-hcc-toolchain.cmake")
endif()
string(REPLACE ";" "::" EP_DEVICE_CANN_PACKAGES "${DEVICE_CANN_PACKAGES}")
string(REPLACE ";" "::" EP_CANN_BINARY_PACKAGES "${CANN_BINARY_PACKAGES}")
string(REPLACE ";" "::" EP_DEVICE_CANN_DEPEND_PACKAGES "${DEVICE_CANN_DEPEND_PACKAGES}")

if(SUPERBUILD_ENABLE_DEVICE)
    # 以 ExternalProject 启动独立 CMake 进程，使用 aarch64 工具链交叉编译 device 侧，
    # 编译完成后执行 cpack 打包，产物为 device-<pkg>.tar.gz。
    include(ExternalProject)
    ExternalProject_Add(cann_device
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device
        BINARY_DIR ${CMAKE_BINARY_DIR}/device_build
        CMAKE_ARGS
            -DENABLE_OPEN_SRC=TRUE
            -DENABLE_UNIFIED_BUILD=TRUE
            -DBUILD_OPEN_PROJECT=TRUE
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCANN_SUPERBUILD_CONFIG=${CANN_SUPERBUILD_CONFIG}
            -DCMAKE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN_FILE}
            -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
            -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}
            -DCANN_PACKAGES=${EP_DEVICE_CANN_PACKAGES}
            -DCANN_BINARY_PACKAGES=${EP_CANN_BINARY_PACKAGES}
            -DCANN_DEPEND_PACKAGES=${EP_DEVICE_CANN_DEPEND_PACKAGES}
            -DTOOLCHAIN_DIR=${DEVICE_TOOLCHAIN_DIR}
            -DENABLE_SIGN=${ENABLE_SIGN}
            -DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT}
            -DLAUNCH_COMPILE_TOOL=${LAUNCH_COMPILE_TOOL}
            -DLAUNCH_LINK_TOOL=${LAUNCH_LINK_TOOL}
        INSTALL_COMMAND ${CMAKE_CPACK_COMMAND}
        LIST_SEPARATOR ::
        BUILD_ALWAYS TRUE
    )
    # 在 configure 前创建 CMake File API 的 codemodel-v2 查询文件，
    # 使 CMake 写出 codemodel 响应，供 IDE/工具感知 device 构建结构。
    ExternalProject_Add_Step(cann_device query_codemodel
 	    COMMAND ${CMAKE_COMMAND} -E make_directory <BINARY_DIR>/.cmake/api/v1/query
 	    COMMAND ${CMAKE_COMMAND} -E touch <BINARY_DIR>/.cmake/api/v1/query/codemodel-v2
 	    DEPENDEES patch
 	    DEPENDERS configure
 	)
    # 将 device 侧产物 device-<pkg>.tar.gz 收集到对应组件，供 host 侧打包流程统一组装。
    foreach(CANN_PACKAGE IN LISTS DEVICE_CANN_PACKAGES)
        install(FILES
            ${CMAKE_BINARY_DIR}/device_build/device-${CANN_PACKAGE}.tar.gz
            DESTINATION .
            COMPONENT ${CANN_PACKAGE}
        )
    endforeach()
endif()

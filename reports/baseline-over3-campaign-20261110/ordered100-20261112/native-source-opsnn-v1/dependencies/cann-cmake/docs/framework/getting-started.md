<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# 子包接入与独立构建

本文档面向 CANN 子包开发者，说明如何引入 cmake 框架并获得公共编译能力，以及如何独立构建子包。

## 适用场景

- 子包开发阶段，只需编译当前包，不需要联合编译其它组件
- 调试单个组件的编译问题
- CI 中对单个包进行验证

子包独立构建时，`TOPLEVEL_PROJECT` 为 `ON`，打包、device 交叉编译、依赖检查等功能生效。

## 接入步骤

### 1. 创建 fetch_cann_cmake.cmake

在子包仓库的 `cmake/fetch_cann_cmake.cmake` 中粘贴以下代码，通过 FetchContent 拉取本仓库：

```cmake
if(NOT PROJECT_SOURCE_DIR)
    set(CANN_CMAKE_TAG "master-025")
    if(CANN_3RD_LIB_PATH AND IS_DIRECTORY "${CANN_3RD_LIB_PATH}/cann-cmake")
        include("${CANN_3RD_LIB_PATH}/cann-cmake/function/prepare.cmake")
    else()
        include(FetchContent)

        if(CANN_3RD_LIB_PATH AND EXISTS "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz")
            FetchContent_Declare(
                cann-cmake
                URL "${CANN_3RD_LIB_PATH}/cmake-${CANN_CMAKE_TAG}.tar.gz"
                URL_HASH SHA256=<填入实际哈希>
            )
        else()
            FetchContent_Declare(
                cann-cmake
                GIT_REPOSITORY https://gitcode.com/cann/cmake.git
                GIT_TAG        ${CANN_CMAKE_TAG}
                GIT_SHALLOW    TRUE
            )
        endif()
        FetchContent_GetProperties(cann-cmake)
        if(NOT cann-cmake_POPULATED)
            FetchContent_Populate(cann-cmake)
        endif()
        include("${cann-cmake_SOURCE_DIR}/function/prepare.cmake")
    endif()
endif()
```

> `if(NOT PROJECT_SOURCE_DIR)` 判断确保：当子包被 superbuild 通过 `add_subdirectory` 引入时（`PROJECT_SOURCE_DIR` 已设置），跳过 FetchContent，复用 superbuild 已加载的框架。

### 2. 在 CMakeLists.txt 中引入并初始化

```cmake
cmake_minimum_required(VERSION 3.16)
include(cmake/fetch_cann_cmake.cmake)
project(runtime)

init_cann_project()
```

关键顺序：`include` 在 `project()` 之前，`init_cann_project()` 在 `project()` 之后。

- **`include` 在 `project()` 之前**：框架用 `PROJECT_SOURCE_DIR` 是否为空来区分独立构建和多仓联编。`project()` 之前该变量为空，框架识别为独立构建；若放在 `project()` 之后，该变量已被设置，框架会误判为多仓联编，导致打包、device 构建、依赖检查等功能被跳过。
- **`init_cann_project()` 在 `project()` 之后**：该函数设置 C++17、ccache 等，需要 `project()` 初始化的内置变量（如 `CMAKE_SOURCE_DIR`、`CMAKE_SYSTEM_PROCESSOR`）已就绪。

### 3. 调用公共 API

初始化后，即可在 CMakeLists.txt 中调用框架提供的公共 API，参见 [public-api.md](public-api.md)。

## init_cann_project() 做了什么

`init_cann_project()` 执行以下初始化：

| 操作 | 说明 |
|------|------|
| 设置 C++17 标准 | `CMAKE_CXX_STANDARD 17`，`CMAKE_CXX_EXTENSIONS OFF` |
| 启用 ccache | `ENABLE_CCACHE` 默认 `TRUE`，自动检测并设置为编译器启动器 |
| `__FILE__` 宏改写 | 非 Ninja 模式下，`__FILE__` 只输出文件名而非完整路径 |
| 设置安装路径变量 | `INSTALL_LIBRARY_DIR`、`INSTALL_RUNTIME_DIR` 等 |
| 解析 CANN 工具包路径 | `ASCEND_INSTALL_PATH` ← `ASCEND_CANN_PACKAGE_PATH` |
| 设置第三方库默认路径 | `CANN_3RD_LIB_PATH` 默认为 `${CMAKE_SOURCE_DIR}/third_party` |
| 添加 CMake 模块路径 | `PREPEND_MODULE_PATH` 参数将 `modules/` 加入 `CMAKE_MODULE_PATH` |
| 检测目标架构 | `TARGET_ARCH` 设为 `x86_64` 或 `aarch64` |

该宏有幂等保护（`CANN_PROJECT_INITED`），联合构建时多次调用只在首次生效。

## 环境变量

| 变量 | 必选 | 说明 |
|------|------|------|
| `ASCEND_CANN_PACKAGE_PATH` | 是 | CANN 工具包安装路径（如 `/usr/local/Ascend/cann`） |
| `CANN_3RD_LIB_PATH` | 否 | 第三方库路径，默认为 `${CMAKE_SOURCE_DIR}/third_party` |
| `ENABLE_OPEN_SRC` | 否 | 开源构建模式标记，设为 `TRUE` |

## 构建命令

```bash
# 配置
cmake -B build \
    -DENABLE_OPEN_SRC=TRUE \
    -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann

# 编译
cmake --build build -j8

# 安装（可选）
cmake --install build --prefix /path/to/install
```

通过 `CMAKE_BUILD_TYPE` 指定构建类型：

```bash
# Release（默认）
cmake -B build -DCMAKE_BUILD_TYPE=Release ...

# Debug
cmake -B build -DCMAKE_BUILD_TYPE=Debug ...
```

常用选项：

| CMake 变量 | 说明 | 默认值 |
|-----------|------|--------|
| `ENABLE_OPEN_SRC` | 开源构建模式 | `FALSE` |
| `ENABLE_CCACHE` | 启用 ccache | `TRUE` |
| `ENABLE_ASAN` | 启用 AddressSanitizer | `FALSE` |
| `ENABLE_GCOV` | 启用代码覆盖率 | `FALSE` |
| `ENABLE_STRICT_WARNINGS` | 启用增强告警（须在首次 include `intf_pub` 前设置） | `FALSE` |
| `USE_CXX11_ABI` | 覆盖 CXX11 ABI 设置（host 默认 `0`，device 默认 `1`） | 自动 |
| `CMAKE_BUILD_TYPE` | 构建类型 | `Release` |

## host 侧与 device 侧构建

子包默认仅构建 host 侧。如果涉及 device 侧编译，可按以下步骤添加 device 构建。

### 添加 device 入口（可选）

如果子包涉及 device 侧编译，可在子仓根目录下创建 `cmake/device/` 子目录作为 device 构建入口，并在子仓的 CMakeLists.txt 中调用 `add_cann_device_project(component)` 注册 device 工程。该函数会通过 ExternalProject 启动独立的交叉编译流程，引入 `cmake/device/` 下的 CMakeLists.txt。

### 触发 device 构建

`ENABLE_BUILD_DEVICE` 控制 device 构建是否执行。默认开启，通过 `-DENABLE_BUILD_DEVICE=FALSE` 可跳过 device 构建。

```bash
cmake -B build \
    -DENABLE_OPEN_SRC=TRUE \
    -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann \
    -DENABLE_BUILD_DEVICE=TRUE

cmake --build build -j8
```

### host 与 device 构建的区别

| 方面 | host | device |
|------|------|--------|
| 工具链 | 系统默认 GCC | hcc 交叉编译工具链 |
| `PRODUCT_SIDE` | `host`（或不设置） | `device` |
| ABI | `_GLIBCXX_USE_CXX11_ABI=0` | `_GLIBCXX_USE_CXX11_ABI=1` |
| 源码目录 | `<pkg>/` | `<pkg>/cmake/device/` |

## 环境依赖

| 依赖 | 说明 |
|------|------|
| CMake >= 3.16.3 | |
| GCC >= 7.3.0 | 支持 C++17 |
| Python 3.7+ | 部分构建脚本依赖 |
| ccache（可选） | 默认启用，加速增量编译 |

如需多仓联合编译，参见 [superbuild/getting-started.md](../superbuild/getting-started.md)。

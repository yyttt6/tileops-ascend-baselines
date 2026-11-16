<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# 整体架构

## 仓库定位

cmake 仓库是 CANN 生态的公共构建框架，为所有 CANN 组件仓（runtime、metadef、ge 等）提供统一的编译、打包、安装能力。它本身不产出可运行制品，而是作为构建基础设施被各组件仓引用。

## 模块分层

```
┌─────────────────────────────────────────────────────────────┐
│                    CANN 组件生态                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ runtime  │  │ metadef  │  │  hcomm   │  │   ...    │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │             │             │             │           │
│       └─────────────┴──────┬──────┴─────────────┘           │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │      cmake      │  ← 本仓库              │
│                   └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

本仓库内部按职责分为四层：

### 构建入口层

| 路径 | 职责 |
|------|------|
| `build.sh` | 多仓联合编译构建入口脚本，解析参数后调用 cmake 配置和编译 |
| `superbuild/CMakeLists.txt` | superbuild 的 cmake 入口脚本，调度 host 与 device 构建，协调多仓联合编译 |
| `superbuild/device/CMakeLists.txt` | device 侧 superbuild 工程，通过 ExternalProject 触发 |
| `superbuild/config.cmake` | 包名到源码目录的映射、组件定义 |

### 框架核心层

| 路径 | 职责 |
|------|------|
| `function/prepare.cmake` | 公共 API 定义，被组件仓和 superbuild 共同引用 |
| `function/function.cmake` | superbuild 初始化、依赖解析、目标过滤 |
| `intf_pub/intf_pub_linux.cmake` | 共享编译/链接选项：安全加固、sanitizer、ABI |
| `toolchain/aarch64-hcc-toolchain.cmake` | device 交叉编译工具链（aarch64 via hcc） |
| `modules/Find<dep>.cmake` | CMake find 模块，通过 `PREPEND_MODULE_PATH` 加入搜索路径 |

### 第三方依赖层

| 路径 | 职责 |
|------|------|
| `third_party/<name>.cmake` | 第三方库构建脚本（abseil、boost、grpc、protobuf 等），通过 `add_cann_third_party(name)` 引入 |

### 脚本工具层

| 路径 | 职责 |
|------|------|
| `scripts/package/` | Python 打包逻辑 + pytest 测试，CMake `package` 阶段调用 |
| `scripts/version/` | 版本信息生成、构建依赖检查 |
| `scripts/sign/` | 代码签名脚本 |
| `scripts/signtool/` | 镜像打包/提取/ESBC 头工具 |
| `scripts/install/` | 安装脚本，随包发布 |
| `scripts/build_analysis/` | IWYU 日志解析器，头文件依赖分析 |

## 核心概念

### CANN_TOP_DIR

所有 CANN 组件仓的共同父目录。在本仓库中通过相对路径自动解析：

- host 侧：`CANN_TOP_DIR` = cmake 仓库的父目录
- device 侧：因 device 构建从 `superbuild/device/` 出发，需多回溯一层，`CANN_TOP_DIR` = cmake 仓库的父目录的父目录

包依赖解析时，从 `${CANN_TOP_DIR}/${pkg_dir}/version.cmake` 读取各包的版本和依赖声明。包名到目录的映射定义在 `superbuild/config.cmake`。

### 两种集成模式

框架通过 `TOPLEVEL_PROJECT` 变量区分两种使用场景：

| 模式 | TOPLEVEL_PROJECT | 触发条件 | 行为 |
|------|-----------------|----------|------|
| 单仓编译 | `ON` | 组件仓 `CMakeLists.txt` 中 `include(fetch_cann_cmake.cmake)` 在 `project()` 之前，`PROJECT_SOURCE_DIR` 为空 | 打包、device 构建、依赖检查等功能直接生效 |
| 多仓联编 | `OFF` | `superbuild/CMakeLists.txt` 中 `project()` 在 `include` 之前，`PROJECT_SOURCE_DIR` 已设置 | 部分函数（打包、device 构建、依赖检查）早返回，由 superbuild 统一管理；`ENABLE_UNIFIED_BUILD` 可重新启用 |

判定依据是 `include(prepare.cmake)` 时 `PROJECT_SOURCE_DIR` 是否已设置：为空说明尚无 `project()` 调用，处于单仓编译模式；已设置说明 `project()` 已调用，处于多仓联编模式。

`ENABLE_UNIFIED_BUILD` 可在多仓联编模式（`TOPLEVEL_PROJECT OFF`）下重新启用部分功能。

### host/device 双构建

CANN 组件分为主机侧（host）和设备侧（device）两部分，构建流程分离：

- **host 构建**：直接在 `superbuild/CMakeLists.txt` 中 `add_subdirectory` 引入各组件仓的主 CMakeLists.txt
- **device 构建**：通过 `ExternalProject_Add` 启动独立 CMake 进程，使用 `toolchain/aarch64-hcc-toolchain.cmake` 交叉编译，引入各组件仓的 `cmake/device/` 子目录

`PRODUCT_SIDE` 变量（`host` / `device`）控制当前构建侧，影响源码目录选择和编译选项。

### 依赖解析

`build.sh --pkgs=runtime,asc-devkit` 指定要构建的包。框架通过递归解析各包 `version.cmake` 中的构建依赖声明，自动拉入所有依赖包：

```
get_pkg_dependencies("runtime")
  → 读取 ${CANN_TOP_DIR}/runtime/version.cmake
  → 获取该包的构建依赖列表
  → 对每个依赖包递归解析
  → 结果存入 CANN_DEPEND_PACKAGES
```

详细的解析过程和示例参见 [superbuild/internals.md](superbuild/internals.md)。

### 公共 API

`function/prepare.cmake` 中的函数是框架的公共 API，被组件仓的 CMakeLists.txt 调用：

| 函数 | 用途 |
|------|------|
| `init_cann_project()` | 初始化工程，设置 C++17、ccache、模块路径等 |
| `set_cann_package(name)` | 声明包名和版本号 |
| `set_cann_build_dependencies(pkg dep)` | 声明构建依赖 |
| `set_cann_run_dependencies(pkg dep)` | 声明运行依赖 |
| `set_cann_cpack_config(component)` | 配置 CPack 打包参数 |
| `add_cann_third_party(name)` | 引入第三方库构建脚本 |
| `find_cann_package(name)` | 包装的 find_package，联编时跳过 |
| `generate_cann_stub_library(name)` | 生成打桩库 |
| `cann_pack_targets_and_files()` | 打包目标文件和普通文件 |
| `gen_cann_version_header(pkg)` | 生成版本号头文件 |
| `add_cann_sign_file()` | 添加代码签名 |

修改这些函数的签名或行为是跨仓破坏性变更。

### 编译选项约定

`intf_pub/intf_pub_linux.cmake` 定义了全局编译选项：

- **安全加固**：`-fstack-protector-strong`、`-fPIC`、`-Wall -Wextra`
- **Sanitizer**：ASan/TSan/UBSan，通过 `ENABLE_ASAN`/`ENABLE_TSAN`/`ENABLE_UBSAN` 控制
- **严格告警**：`-Wformat=2`/`-Wduplicated-branches`/`-Wlogical-op` 等，通过 `ENABLE_STRICT_WARNINGS` 控制（默认关闭）。须在首次 include `intf_pub` 之前设置方可生效
- **ABI**：`CANN_CXX11_ABI` 在 `init_cann_project()` 中统一计算（`function/prepare.cmake`），供 `intf_pub` 及三方库共用。host 默认 `_GLIBCXX_USE_CXX11_ABI=0`（旧 ABI），device 默认 `1`，`USE_CXX11_ABI` 可覆盖
- **`__FILE__` 宏改写**：非 Ninja 模式下，编译规则被覆写为只输出文件名（不输出完整路径），同时添加 `-Wno-builtin-macro-redefined`。device 工具链文件中同样覆写

### C++ 标准

C++17（`CMAKE_CXX_STANDARD 17`，`CMAKE_CXX_EXTENSIONS OFF`），在 `init_cann_project` 中设置。

### ccache

默认启用（`ENABLE_CCACHE` 默认 `TRUE`），自动检测 `ccache` 并设置为编译器启动器。

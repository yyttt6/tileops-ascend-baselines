<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# superbuild 构建机制

本文档介绍 superbuild 工程的设计原理与工作流程。使用指南参见 [getting-started.md](getting-started.md)。

## 1. 设计原理

### 1.1 问题背景

CANN 生态由多个独立组件仓组成（runtime、metadef、ge 等），各仓可独立编译，但联合构建时存在跨仓依赖关系。如果让每个仓各自独立编译，存在以下问题：

- 依赖顺序需要人工维护
- 重复编译依赖项
- host/device 产物需要分别管理
- 打包需要跨仓收集产物

### 1.2 设计目标

superbuild 工程作为"元构建"（meta-build）层，解决以下问题：

1. **自动依赖解析**：根据 `--pkgs` 参数和各仓 `version.cmake` 中的依赖声明，自动计算完整的构建依赖链
2. **统一编译入口**：一个 `build.sh` 命令完成所有组件的编译和打包
3. **host/device 分离**：host 侧直接编译，device 侧通过 ExternalProject 交叉编译
4. **产物归集**：自动收集各组件的安装产物，统一打包

### 1.3 核心设计决策

| 决策 | 原因 |
|------|------|
| host 侧使用 `add_subdirectory` 引入各仓（而非 ExternalProject） | 各仓共享同一 CMake 上下文，目标间可直接引用 |
| device 侧使用 ExternalProject | 需要交叉编译，独立的 CMake 进程和工具链 |
| 通过 `version.cmake` 声明依赖 | 依赖信息与组件版本绑定，避免中心化维护 |
| `cann_all_targets` 聚合目标 | 通过 `EXCLUDE_FROM_ALL` 精确控制只编译指定包及其依赖 |

## 2. 文件结构

```
superbuild/
├── CMakeLists.txt      # host 侧 superbuild 入口
├── config.cmake        # 包名→源码目录映射、组件定义
├── device/
│   └── CMakeLists.txt  # device 侧 superbuild 入口
└── libraries/
    └── .gitkeep        # 占位，预留给外部库存放
```

### 2.1 包映射（config.cmake）

定义两个映射关系：

**包名 → 源码目录：** 将 `--pkgs` 指定的包名映射到 `CANN_TOP_DIR` 下的实际源码目录。未在映射中定义的包名，默认使用包名本身作为目录名。

**包 → 组件列表：** 定义每个包包含的组件，用于打包时区分 `COMPONENT`。一个包可以包含多个组件。

### 2.2 自定义配置

`build.sh --superbuild-config=<PATH>` 可指定额外的配置文件，在默认 `config.cmake` 之后加载，用于覆盖默认映射或添加新包。

## 3. host 侧构建流程

host 侧的 superbuild 构建按以下步骤进行：

### 步骤 1：初始化

调用 `init_cann_superbuild_project`（`PRODUCT_SIDE` 设为 `"host"`），完成：

1. 解析 `CANN_CMAKE_DIR`（本仓库根目录）和 `CANN_TOP_DIR`（本仓库的父目录，即组件仓的共同父目录）
2. 调用 `init_cann_project()` 设置公共配置：C++17 标准、ccache、`CMAKE_MODULE_PATH` 等
3. 加载 `superbuild/config.cmake`（包映射）
4. 计算二进制组件列表

### 步骤 2：设置全局变量

superbuild 在初始化后设置若干全局变量，其中最核心的是控制**两种构建模式的开关变量**。

#### 两种构建模式

框架通过 `TOPLEVEL_PROJECT` 变量区分两种模式：

| 模式 | 触发条件 | `TOPLEVEL_PROJECT` | 典型场景 |
|------|---------|-------------------|---------|
| 单仓编译 | `include(prepare.cmake)` 时 `PROJECT_SOURCE_DIR` 为空 | `ON` | 组件仓独立编译 |
| 多仓联编 | `include(prepare.cmake)` 时 `PROJECT_SOURCE_DIR` 已设置 | `OFF` | superbuild 联合构建 |

`prepare.cmake` 中的判定逻辑：

```cmake
if(NOT PROJECT_SOURCE_DIR)
    set(TOPLEVEL_PROJECT ON)    # 单仓编译
else()
    set(TOPLEVEL_PROJECT OFF)   # 多仓联编
endif()
```

#### `TOPLEVEL_PROJECT` 的作用

`TOPLEVEL_PROJECT` 控制框架公共 API 函数是否执行。多仓联编时（`OFF`），部分函数会 early-return 以避免与 superbuild 的集中管理冲突：

| 函数 | `TOPLEVEL_PROJECT OFF` 时行为 | 原因 |
|------|-------------------------------|------|
| `set_cann_cpack_config` | 直接返回 | 打包由 superbuild 统一管理 |
| `add_cann_device_project` | 直接返回 | device 构建由 superbuild 的 ExternalProject 统一管理 |
| `check_cann_pkg_build_deps` | 直接返回 | 依赖检查由 superbuild 统一执行 |
| `set_cann_subprj_package` | 直接返回 | 子工程打包由 superbuild 统一管理 |
| `add_cann_third_party` | 仅 `ENABLE_UNIFIED_BUILD` 时执行 | 联编时三方库由 superbuild 统一引入 |
| `find_cann_package` | 仅二进制组件调用 `find_package`，其余跳过 | 目标已在同一 CMake 上下文中 |
| `init_cann_project` | 正常执行 | 幂等保护，多次调用只在首次生效 |
| `set_cann_package` | 正常执行 | 版本声明在任何模式下都需要 |
| `set_cann_build_dependencies` | 正常执行 | 依赖声明在任何模式下都需要 |
| `set_cann_run_dependencies` | 正常执行 | 依赖声明在任何模式下都需要 |

#### `ENABLE_UNIFIED_BUILD`

`ENABLE_UNIFIED_BUILD` 用于在 `TOPLEVEL_PROJECT OFF` 时重新启用部分功能，适用于多仓统一构建场景（如 device 侧构建）。其效果：

- `set_cann_cpack_config`、`add_cann_third_party`、`find_cann_package` 等函数在 `ENABLE_UNIFIED_BUILD ON` 时恢复执行
- 但 `set_cann_cpack_config` 会额外检查组件是否在 `CANN_PACKAGES` 列表中，只对目标包生效

简言之：`TOPLEVEL_PROJECT` 表示"当前是否为顶层独立项目"，`ON` 为单仓独立编译，`OFF` 为多仓联编（superbuild 主导）；`ENABLE_UNIFIED_BUILD` 是在多仓联编下"子工程是否参与统一构建"的分开关。

#### 其它全局变量

| 变量 | 说明 |
|------|------|
| `BUILD_OPEN_PROJECT` | 标记开源构建模式，透传到 device 构建 |
| `ENABLE_PACKAGE` | 默认开启打包 |

### 步骤 3：依赖解析

#### 依赖声明格式

每个组件仓在自身的 `version.cmake` 中通过 `set_cann_build_dependencies` 声明构建依赖。例如 `hcomm/version.cmake`：

```cmake
set_cann_package(hcomm VERSION "9.1.0")

set_cann_build_dependencies(runtime "CUR_MAJOR_MINOR_VER")
set_cann_build_dependencies(metadef "CUR_MAJOR_MINOR_VER")
set_cann_build_dependencies(bisheng-compiler "CUR_MAJOR_MINOR_VER")
set_cann_build_dependencies(asc-devkit "CUR_MAJOR_MINOR_VER")
```

每条 `set_cann_build_dependencies(dep "version_spec")` 将依赖包名和版本约束以**成对**形式追加到 `CANN_VERSION_<pkg>_BUILD_DEPS` 列表，形如 `"runtime;9.1;metadef;9.1"`。解析时按步长 2 提取包名即可。

#### 递归解析过程

`get_pkg_dependencies` 调用 `do_get_pkg_dependencies` 进行递归广度优先解析，算法如下：

1. 对当前层的每个包，读取其 `version.cmake`，提取构建依赖列表
2. 对每个依赖包，检查跳过条件：
   - 已在结果列表 `all_pkgs` 中（避免重复、避免环）
   - 已在待处理队列 `pkgs_next` 中（避免重复入队）
   - 属于二进制包 `CANN_BINARY_PACKAGES`（二进制包不编译，直接使用预编译产物）
   - `bisheng-compiler`（特殊处理，不参与源码编译）
3. 未跳过的依赖加入下一层待处理队列 `pkgs_next`
4. 将 `pkgs_next` 并入结果列表，对 `pkgs_next` 递归执行，直到没有新的依赖

#### 示例

假设组件仓的依赖关系如下：

```
runtime          (无依赖)
metadef          → runtime
asc-devkit       → runtime
ge-executor      → metadef, runtime
hcomm            → runtime, metadef, bisheng-compiler, asc-devkit
```

执行 `sh build.sh --pkgs=hcomm` 时的解析过程：

```
初始:  CANN_PACKAGES = [hcomm]
       all_pkgs      = [hcomm]

第 1 层: 处理 [hcomm]
  读取 hcomm/version.cmake → BUILD_DEPS = [runtime, metadef, bisheng-compiler, asc-devkit]
  runtime          → 未跳过，加入 pkgs_next
  metadef          → 未跳过，加入 pkgs_next
  bisheng-compiler → 跳过（特殊处理）
  asc-devkit       → 未跳过，加入 pkgs_next
  结果: all_pkgs = [hcomm, runtime, metadef, asc-devkit]
        pkgs_next = [runtime, metadef, asc-devkit]

第 2 层: 处理 [runtime, metadef, asc-devkit]
  runtime     → 无依赖
  metadef     → BUILD_DEPS = [runtime]  → runtime 已在 all_pkgs，跳过
  asc-devkit  → BUILD_DEPS = [runtime]  → runtime 已在 all_pkgs，跳过
  结果: pkgs_next 为空，递归结束

最终: CANN_DEPEND_PACKAGES = [hcomm, runtime, metadef, asc-devkit]
```

解析完成后，`CANN_DEPEND_PACKAGES` 包含目标包及其所有间接依赖，后续步骤据此引入源码目录、设置构建目标。

#### device 包筛选

**`calc_device_packages`**：遍历 `CANN_DEPEND_PACKAGES`，将所有有 `cmake/device/` 子目录的包加入 `DEVICE_CANN_DEPEND_PACKAGES`；其中属于目标包（在 `CANN_PACKAGES` 中）的再额外加入 `DEVICE_CANN_PACKAGES`。即 `DEVICE_CANN_PACKAGES` ⊆ `DEVICE_CANN_DEPEND_PACKAGES`。`ge-executor`/`ge-compiler` 是纯 host 侧包（与 `dflow-executor` 共享 `ge` 仓源码目录），在遍历时直接跳过，不会出现在两个列表中。这两个列表用于步骤 7 决定是否启动 device 交叉编译。

接上例，若 `runtime` 和 `hcomm` 有 `cmake/device/` 目录，`metadef` 和 `asc-devkit` 没有：

```
DEVICE_CANN_DEPEND_PACKAGES = [hcomm, runtime]  (所有有 device 目录的包)
DEVICE_CANN_PACKAGES        = [hcomm]            (其中属于目标包的)
```

### 步骤 4：device 构建开关

device 构建需同时满足：`CANN_SUPERBUILD_BUILD_DEVICE` 为真（用户未指定 `--build_host_only`），且 `DEVICE_CANN_PACKAGES` 非空（存在需要 device 编译的包）。`ge-executor`/`ge-compiler` 已在 `calc_device_packages` 中排除，不会误触发 device 构建。当用户直接调用 cmake（未经 `build.sh`）时，`ENABLE_BUILD_DEVICE` 未定义，默认按启用处理。

### 步骤 5：引入依赖包

将所有依赖包的源码目录通过 `add_subdirectory` 引入当前 CMake 工程。这里**不能**使用 `EXCLUDE_FROM_ALL`，因为 CMake 会因此排除 `install` 命令，导致依赖包的安装产物缺失。

### 步骤 6：引入目标包并设置构建目标

#### 问题

步骤 5 通过 `add_subdirectory` 引入了所有依赖包，步骤 6 引入目标包。此时整个 CMake 工程中包含了**所有包**的全部 target。CMake 默认将这些 target 都纳入 `all` 目标，执行 `cmake --build build` 会编译**所有包的所有 target**，包括依赖包中目标包根本不需要的 target（如测试目标、其它组件目标）。

#### 解决方案：`set_cann_all_targets`

核心思路分三步：

**第一步：收集目标包的构建目标**

扫描 `--pkgs` 指定包的源码目录，递归收集所有"真实"构建目标。过滤掉 `INTERFACE_LIBRARY`（无编译产物）和已标记 `EXCLUDE_FROM_ALL` 的目标：

```cmake
# get_build_targets_in_directory: 递归扫描目录下的所有 target，过滤掉
# INTERFACE_LIBRARY 和 EXCLUDE_FROM_ALL 的目标
foreach(pkg_dir IN LISTS pkg_dirs)
    get_build_targets_in_directory(pkg_targets ${CANN_TOP_DIR}/${pkg_dir})
    list(APPEND build_targets ${pkg_targets})
endforeach()
```

**第二步：收集整个构建树的所有目标**

从 superbuild 工程根目录出发，递归扫描所有通过 `add_subdirectory` 引入的子目录，收集全部"真实"构建目标（包含目标包、依赖包、superbuild 自身的目标）：

```cmake
get_build_targets_in_directory(all_targets "${CMAKE_SOURCE_DIR}")
```

**第三步：全部标记 EXCLUDE_FROM_ALL，再创建聚合目标**

将第二步收集到的**所有**目标标记为 `EXCLUDE_FROM_ALL`，使其退出默认 `all` 目标。然后创建 `cann_all_targets` 伪目标（`ALL` 属性，仍在 `all` 中），使其只依赖第一步收集的目标包构建目标：

```cmake
# 所有目标退出 all，防止冗余编译
foreach(target IN LISTS all_targets)
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
endforeach()

# cann_all_targets 留在 all 中，只拉起目标包的 target
add_custom_target(cann_all_targets ALL)
if(build_targets)
    add_dependencies(cann_all_targets ${build_targets})
endif()
```

#### 为什么有效

关键在于 CMake 对 `EXCLUDE_FROM_ALL` 的语义：该属性只将目标移出默认 `all` 目标，**并不阻止它作为其他目标的依赖被编译**。

执行 `cmake --build build` 时的构建链：

```
all (默认目标)
 └── cann_all_targets          ← ALL 属性，留在 all 中
      ├── target_pkg_target_a   ← 通过 add_dependencies 拉起
      ├── target_pkg_target_b
      │    └── (link 依赖) dep_pkg_target_x  ← 链接依赖，自动编译
      └── target_pkg_target_c

dep_pkg_target_y  ← EXCLUDE_FROM_ALL 且无目标依赖它，不编译
dep_pkg_target_z  ← 同上，不编译
```

- **目标包的 target**：虽被标记 `EXCLUDE_FROM_ALL`，但被 `cann_all_targets` 通过 `add_dependencies` 显式依赖，因此会被编译
- **依赖包中被链接的 target**：作为目标包 target 的 `target_link_libraries` 依赖，自动被编译
- **依赖包中未被链接的 target**：`EXCLUDE_FROM_ALL` 且无人依赖，不被编译

这样 `cmake --build build` 只会编译 `--pkgs` 指定包的目标及其链接依赖，不会冗余编译依赖包中的无关目标。

#### 目录扫描细节

`get_build_targets_in_directory` 在递归扫描子目录时，会跳过设置了目录级 `EXCLUDE_FROM_ALL` 的子目录。这使得组件仓可以通过在自身 CMakeLists.txt 中对测试等目录设置 `EXCLUDE_FROM_ALL` 来排除其 target 被扫描收集。

### 步骤 7：device 交叉编译

当 device 构建开启时，通过 `ExternalProject_Add` 启动 device 子构建：

- **列表传递**：ExternalProject 不支持 CMake 列表（分号分隔），需将列表转换为 `::` 分隔的字符串，在 device 侧再转回列表
- **工具链**：指定 `aarch64-hcc-toolchain.cmake` 进行交叉编译
- **参数透传**：host 侧的构建参数通过 `CMAKE_ARGS` 传递到 device 构建
- **codemodel 查询**：在 configure 之前创建 codemodel-v2 查询文件，供 IDE 获取构建目标信息
- **产物归集**：device 侧的打包产物（`device-<pkg>.tar.gz`）通过 `install(FILES ...)` 注册到 host 侧的对应组件，host 侧打包时自动包含

## 4. device 侧构建流程

device 侧的 superbuild 入口结构与 host 侧类似但更简洁，主要区别：

| 方面 | host 侧 | device 侧 |
|------|---------|-----------|
| `PRODUCT_SIDE` | `"host"` | `"device"` |
| 源码目录后缀 | 无（直接引入 `<pkg>`） | `cmake/device`（引入 `<pkg>/cmake/device`） |
| 打包方式 | `set_cann_cpack_config` 各组件独立配置 | `set_cann_subprj_package(SUPERBUILD)` 统一打包 |
| 产物名 | `cann-<component>_<version>_<platform>-<arch>.run` | `device-<pkg>.tar.gz` |

device 侧通过 `get_package_dirs` 的第三个参数 `"cmake/device"` 指定子目录后缀，使引入的源码目录变为 `<pkg>/cmake/device`。

## 5. host/device 构建关系

```
build.sh --pkgs=runtime
    │
    ▼
cmake -S superbuild -B build         ← host 侧配置
    │
    ├── init_cann_superbuild_project(PRODUCT_SIDE "host")
    ├── get_pkg_dependencies("runtime") → CANN_DEPEND_PACKAGES
    ├── calc_device_packages() → DEVICE_CANN_PACKAGES
    ├── add_subdirectory(${CANN_TOP_DIR}/runtime)   ← 引入 runtime 源码
    ├── set_cann_all_targets(...)                   ← 设置精确构建目标
    │
    └── [若需要 device 构建]
        ├── ExternalProject_Add(cann_device)        ← 启动 device 子构建
        │   │
        │   ▼
        │   cmake -S superbuild/device -B build/device_build
        │   -DCMAKE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN_FILE}
        │   -DTOOLCHAIN_DIR=${DEVICE_TOOLCHAIN_DIR}
        │       │
        │       ├── init_cann_superbuild_project(PRODUCT_SIDE "device")
        │       ├── add_subdirectory(${CANN_TOP_DIR}/runtime/cmake/device)
        │       └── set_cann_subprj_package(SUPERBUILD)
        │           → 产物: build/device_build/device-runtime.tar.gz
        │
        └── install(FILES .../device-runtime.tar.gz COMPONENT runtime)
                                                         ↑ host 侧打包时归入该组件

cmake --build build --target package   ← 编译并打包
    │
    ├── 编译 cann_all_targets 依赖的 host 目标
    ├── 编译 cann_device (ExternalProject)
    └── cpack 打包 → build_out/cann-runtime_<version>_<platform>-<arch>.run
```

## 6. 关键变量

| 变量 | 来源 | 说明 |
|------|------|------|
| `CANN_PACKAGES` | `build.sh --pkgs` | 用户指定的构建包名列表 |
| `CANN_DEPEND_PACKAGES` | `get_pkg_dependencies()` 计算 | 完整依赖包列表（含间接依赖） |
| `DEVICE_CANN_PACKAGES` | `calc_device_packages()` 计算 | 有 device 目录的目标包（排除 ge-executor/ge-compiler） |
| `DEVICE_CANN_DEPEND_PACKAGES` | `calc_device_packages()` 计算 | 有 device 目录的依赖包（排除 ge-executor/ge-compiler） |
| `CANN_SUPERBUILD_BUILD_DEVICE` | `superbuild/CMakeLists.txt` 保存 | 命令行 `ENABLE_BUILD_DEVICE` 原始值（默认 TRUE） |
| `SUPERBUILD_ENABLE_DEVICE` | host 侧 CMakeLists 判定 | 是否启动 device 构建 |
| `CANN_TOP_DIR` | `init_cann_superbuild_project` 自动解析 | 组件仓的共同父目录 |
| `CANN_CMAKE_DIR` | `init_cann_superbuild_project` 自动解析 | 本仓库根目录 |
| `PRODUCT_SIDE` | `init_cann_superbuild_project` 参数 | `"host"` 或 `"device"` |
| `TOOLCHAIN_DIR` | `${ASCEND_INSTALL_PATH}/toolkit/toolchain/hcc` | hcc 交叉编译工具链路径（device 侧，由 `DEVICE_TOOLCHAIN_DIR` 覆盖） |
| `DEVICE_TOOLCHAIN_DIR` | `build.sh --device-toolchain-dir` | device 侧编译器根目录，未指定时回退到 `TOOLCHAIN_DIR` 默认值 |
| `DEVICE_TOOLCHAIN_FILE` | `build.sh --device-toolchain` | device 侧 `CMAKE_TOOLCHAIN_FILE`，未指定时回退到内置 `aarch64-hcc-toolchain.cmake` |
| `EP_DEVICE_CANN_PACKAGES` | `::` 分隔的字符串 | ExternalProject 透传用 |

## 7. 打包流程

### 7.1 host 侧打包

各组件仓在自身的 CMakeLists.txt 中调用 `set_cann_cpack_config(component)` 配置 CPack 参数。该函数仅在 `TOPLEVEL_PROJECT ON` 或 `ENABLE_UNIFIED_BUILD` 时生效。

支持三种包类型（`build.sh --pkg-type`）：

| 类型 | CPack Generator | 产物 |
|------|----------------|------|
| `run` (默认) | External (makeself) | `.run` 自解压安装包 |
| `rpm` | RPM | `.rpm` 包 |
| `deb` | DEB | `.deb` 包 |

### 7.2 device 侧打包

`set_cann_subprj_package(SUPERBUILD)` 将所有 device 包统一打包为 TGZ，产物为 `device-<pkg>.tar.gz`，由 host 侧注册到对应组件。

## 8. 扩展指南

### 8.1 新增一个构建包

假设要新增名为 `newpkg` 的组件仓：

1. **创建组件仓目录**：在 `CANN_TOP_DIR` 下创建 `newpkg/`，包含 `CMakeLists.txt` 和 `version.cmake`

2. **（可选）添加目录映射**：如果包名与目录名不一致，在 `superbuild/config.cmake` 中添加：
   ```cmake
   set(CANN_PACKAGE_DIR_newpkg "newpkg")  # 若目录名与包名相同则可省略
   ```

3. **（可选）添加组件映射**：如果包包含多个组件：
   ```cmake
   set(CANN_PACKAGE_COMPONENTS_newpkg "comp_a" "comp_b")
   ```

4. **在组件仓的 `version.cmake` 中声明依赖**：
   ```cmake
   set_cann_package(newpkg VERSION "8.0.0")
    set_cann_build_dependencies(runtime "8.0.0")
   ```

5. **构建**：
   ```bash
   sh build.sh --pkgs=newpkg
   ```

### 8.2 新增 device 侧构建

在组件仓中创建 `cmake/device/` 子目录，包含 device 侧的 `CMakeLists.txt`。`calc_device_packages()` 会自动检测到该目录并将其加入 device 构建列表。

### 8.3 调试 superbuild

查看完整的 cmake 配置命令：

```bash
sh build.sh --pkgs=runtime -v
```

查看生成的构建系统：

```bash
# 查看依赖解析结果
grep "CANN_DEPEND_PACKAGES" build/CMakeCache.txt

# 查看 device 构建配置
ls build/device_build/

# 查看目标列表
cmake --build build --target help | grep cann
```

### 8.4 使用自定义 superbuild 配置

```bash
sh build.sh --pkgs=runtime --superbuild-config=/path/to/custom_config.cmake
```

自定义配置文件在 `config.cmake` 之后加载，可覆盖包映射或添加新定义。

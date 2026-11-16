<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# 联合编译使用指南

本文档面向构建/发布工程师，说明如何使用 `build.sh` 进行多仓联合编译。

## 环境准备

### 1. 仓库布局

本仓库与 CANN 组件仓需要位于同一父目录下：

```
CANN_TOP_DIR/                     ← 父目录（CANN_TOP_DIR）
├── cmake/                        ← 本仓库
├── runtime/                      ← CANN runtime 组件
├── metadef/                      ← CANN metadef 组件
├── ge/                           ← CANN ge 组件
└── ...                           ← 其它组件
```

`CANN_TOP_DIR` 在框架初始化时自动解析为本仓库的父目录，无需手动设置。

### 2. CANN 工具包（可选）

仅当涉及 device 交叉编译时需要安装 CANN 工具包，用于提供 hcc 交叉编译工具链。通过 `ASCEND_CANN_PACKAGE_PATH` 指定路径：

- 默认查找顺序：
  1. `--cann_path` / `-p` 参数
  2. `ASCEND_HOME_PATH` 环境变量
  3. `ASCEND_OPP_PATH` 环境变量的父目录
  4. `~/Ascend/cann`（非 root 用户）
  5. `/usr/local/Ascend/cann`（root 用户）

若仅需 host 侧构建（如 `--build_host_only`），无需安装 CANN 工具包。

### 3. 编译工具

- CMake >= 3.16.3
- GCC >= 7.3.0（支持 C++17）
- Python 3.7+
- ccache（可选，默认启用）

## 构建命令

```bash
# --pkgs 为必选参数，逗号分隔包名
sh build.sh --pkgs=<PACKAGES> [-j<N>] [-v] [--pkg-type=run|rpm|deb] [--build-type=Release|Debug]
```

### 示例

```bash
# 构建 runtime 包
sh build.sh --pkgs=runtime

# 构建 runtime 和 asc-devkit，16 线程，verbose 输出
sh build.sh --pkgs=runtime,asc-devkit -j16 -v

# 构建 rpm 包
sh build.sh --pkgs=runtime --pkg-type=rpm

# Debug 构建
sh build.sh --pkgs=runtime --build-type=Debug

# 仅构建 host 侧，跳过 device 交叉编译
sh build.sh --pkgs=runtime --build_host_only
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pkgs=<PACKAGES>` | **必选**，指定构建的包名，逗号分隔 | — |
| `-j<N>` | 编译线程数 | CPU 核数 |
| `-v` / `--verbose` | 显示完整编译命令 | 关闭 |
| `--build-type=<TYPE>` | 构建类型：`Release` / `Debug` | `Release` |
| `--pkg-type=<TYPE>` | 包类型：`run` / `rpm` / `deb` | `run` |
| `--build_host_only` | 仅构建 host 侧，跳过 device 交叉编译 | 关闭 |
| `--asan` | 启用 AddressSanitizer | 关闭 |
| `--cov` | 启用代码覆盖率 | 关闭 |
| `-p` / `--cann_path` | 指定 CANN 工具包安装路径 | 自动查找 |
| `--cann_3rd_lib_path` | 第三方库路径 | `./output/third_party` |
| `--binary-pkgs=<PACKAGES>` | 使用指定包的二进制产物而非编译 | — |
| `--enable-sign` | 启用代码签名 | 关闭 |
| `--sign-script <PATH>` | 指定签名脚本路径 | 内置默认 |
| `--superbuild-config=<PATH>` | 自定义 superbuild 配置文件 | 无 |
| `--rule-launch <TOOL>` | 设置编译器和链接器启动规则 | 无 |
| `--make-profiler <TOOL>` | 设置make阶段性能剖析工具 | 无 |
| `--cmake-extra-args <ARGS>` | 支持向cmake工程传递的额外参数，参数之间以逗号分隔，如`--cmake-extra-args ENABLE_BUILD_DEVICE=OFF,USE_CXX_API=1` | 无 |
| `--host-toolchain=<PATH>` | 指定 host 侧 `CMAKE_TOOLCHAIN_FILE` | 无 |
| `--host-toolchain-dir=<PATH>` | 指定 host 侧 `TOOLCHAIN_DIR`（toolchain file 内部引用的编译器根目录） | 无 |
| `--device-toolchain=<PATH>` | 指定 device 侧 `CMAKE_TOOLCHAIN_FILE` | 内置 `aarch64-hcc-toolchain.cmake` |
| `--device-toolchain-dir=<PATH>` | 指定 device 侧 `TOOLCHAIN_DIR`（toolchain file 内部引用的编译器根目录） | `${ASCEND_INSTALL_PATH}/toolkit/toolchain/hcc` |

## 构建产物

- 构建目录：`build/`
- 打包产物：`build_out/`

产物格式取决于 `--pkg-type`：

| 类型 | 产物格式 |
|------|---------|
| `run`（默认） | `.run` 自解压安装包 |
| `rpm` | `.rpm` 包 |
| `deb` | `.deb` 包 |

## 构建流程

`build.sh` 执行两步：

1. `cmake -S superbuild -B build` — 配置阶段，解析依赖、生成 Makefile
2. `cmake --build build --target package` — 编译并打包

配置阶段会：
- 解析 `--pkgs` 指定的包及其依赖链
- 通过 `add_subdirectory` 引入所有相关组件仓
- 判断是否需要 device 交叉编译

详细机制参见 [internals.md](internals.md)。

## 自定义工具链

内置 `toolchain/aarch64-hcc-toolchain.cmake` 用于 hcc 交叉编译 device 侧。如需使用 clang/LLVM 工具链，可指定 `--device-toolchain` 指向 `toolchain/aarch64-clang-toolchain.cmake`：

```bash
sh build.sh --pkgs=runtime \
  --device-toolchain=cmake/toolchain/aarch64-clang-toolchain.cmake \
  --device-toolchain-dir=/usr/lib/llvm-15
```

`TOOLCHAIN_DIR` 指向 LLVM 安装根目录，其 `bin/` 下应包含 `clang`、`clang++`、`lld`、`llvm-ar` 等工具。

## 可用的包名

包名到源码目录的映射定义在 `superbuild/config.cmake`：

| 包名 | 源码目录 |
|------|---------|
| `npu-runtime` | `runtime` |
| `asc-devkit` | `asc/asc-devkit` |
| `asc-tools` | `asc/asc-tools` |
| `ge-executor` | `ge` |
| `ge-compiler` | `ge` |
| `dflow-executor` | `ge` |

未在映射中定义的包名，默认使用包名本身作为目录名。

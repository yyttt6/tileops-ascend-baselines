<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# CANN cmake 文档

本仓库有两大用途：

1. **为 CANN 子包提供公共编译能力** — 子包通过 `fetch_cann_cmake.cmake` 引入框架，调用公共 API 完成编译、打包、安装。
2. **多仓联合编译** — 以 `superbuild/CMakeLists.txt` 为入口，通过 `build.sh` 驱动，一次性编译多个 CANN 组件并打包。

## 文档导航

### 子包接入与独立构建

→ [framework/getting-started.md](framework/getting-started.md)：接入框架、独立构建子包

→ [framework/public-api.md](framework/public-api.md)：公共 API 参考

### 多仓联合编译

→ [superbuild/getting-started.md](superbuild/getting-started.md)：superbuild 构建用法

→ [superbuild/internals.md](superbuild/internals.md)：superbuild 构建机制与工作流程

### 整体架构

→ [architecture.md](architecture.md)：模块分层、两种集成模式、host/device 双构建

### 安装权限

→ [install/permissions.md](install/permissions.md)：CANN 安装权限规范（安装场景、权限位对照、最小权限原则）

### 辅助工具

→ [scripts/build-analysis.md](scripts/build-analysis.md)：IWYU 头文件依赖分析

→ [scripts/sign-guide.md](scripts/sign-guide.md)：代码签名使用指南

→ [scripts/sign-internals.md](scripts/sign-internals.md)：代码签名实现说明

## 开发本仓库

```bash
# 运行全部测试（在仓库根目录执行）
python3 -m pytest scripts/

# 运行单个测试
python3 -m pytest scripts/package/tests/test_package.py::TestClass -v
```

<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# 代码签名使用指南

> 脚本位置：`scripts/sign/`（`add_header_sign.py`、`community_sign_build.py`）及 `scripts/signtool/`（`esbc_header/`、`image_pack/`、`image_extract/`）

对 CANN 镜像文件制作 CMS 签名并绑定 ESBC 头。实现说明与内部机制参见 [sign-internals.md](sign-internals.md)。

## 功能特性

- **CMS 签名**：调用 signatrust_client 对镜像制作 detached p7s 签名
- **可扩展签名后端**：通过 `--print-sign-ext`/`--print-certtype` 查询 flag 支持自定义签名脚本（产出 `.cms` 等格式、`0x2`/`0xFFFFFFFF` 证书类型），编排器自动查询产物形态
- **ESBC 头绑定**：对配置了 nvcnt 的镜像写入 256 字节 ESBC 二级头
- **镜像加头**：调用 `image_pack.py` 绑定最终文件头（版本、tag、nvcnt、cms 签名、CRL）
- **ini 摘要生成**：调用 `ini_gen.py` 为每个镜像生成 SHA256 摘要文件
- **CRL 转换**：将 PEM 格式 CRL 转为 DER 格式供绑头使用
- **配置驱动**：通过 `bios_check_cfg.xml` 声明每个镜像的签名类型、版本、tag 等属性

## 前置要求

- Python 3.7+
- OpenSSL（CRL 格式转换）
- curl（下载 CRL，仅签名模式需要）
- signatrust_client（CMS 签名，仅签名模式需要）

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `HI_PYTHON` | Python 解释器路径 | 未设置时用 `sys.executable` |
| `SIGN_CLIENT_PATH` | signatrust_client 可执行文件路径 | `/home/jenkins/signatrust_client/signatrust_client` |
| `SIGN_CLIENT_CONFIG` | signatrust_client 配置文件路径 | `/home/jenkins/signatrust_client/client.toml` |
| `CRL_DOWNLOAD_URL` | CRL 下载地址 | 云服务 OBS 地址 |
| `CRL_FILE_PATH` | 本地 CRL 文件路径，设置后跳过远程下载 | 空（每目标独立下载） |
| `SIGN_LOG_LEVEL` | 日志级别（`community_sign_build.py` 专用） | `INFO` |

## 使用方法

### 1. 通过 CMake 集成（推荐）

在组件仓的 `CMakeLists.txt` 中调用 `add_cann_sign_file`：

```cmake
add_cann_sign_file(
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/my_image.bin
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/bios_check_cfg.xml
    RESULT_VAR SIGNED_IMAGE
    DEPENDS my_image_target
)
```

通过 `VERSION` 参数为签名目标指定版本号（必选）：

```cmake
add_cann_sign_file(
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/my_image.bin
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/bios_check_cfg.xml
    RESULT_VAR SIGNED_IMAGE
    DEPENDS my_image_target
    VERSION 1.2.3
)
```

构建时通过 `build.sh` 启用签名：

```bash
# 启用签名
sh build.sh --pkgs=runtime --enable-sign

# 指定自定义签名脚本
sh build.sh --pkgs=runtime --enable-sign --sign-script /path/to/custom_sign.py
```

`add_cann_sign_file` 的完整参数说明参见 [public-api.md](../framework/public-api.md#签名)。

**CMake 层行为**：`add_cann_sign_file` 创建 `add_custom_command`，先拷贝 `INPUT` 到 `${CMAKE_CURRENT_BINARY_DIR}/signatures_${safe_input_name}/`（每目标独立目录），再执行签名命令。生成的签名文件路径通过 `RESULT_VAR` 返回，后续可用于安装或打包。

### 2. 直接调用脚本

不通过 CMake，直接运行签名脚本：

```bash
# 不签名，仅加头（社区开发者常用）
python3 scripts/sign/add_header_sign.py /path/to/images false

# 签名 + 加头
python3 scripts/sign/add_header_sign.py /path/to/images true \
    --bios_check_cfg=bios_check_cfg.xml \
    --version=8.0.0 \
    --sign_script=scripts/sign/community_sign_build.py
```

### 3. 使用自定义签名脚本

自定义签名脚本的生效方式取决于脚本类型：

- **`.sh` 脚本**：通过 `build.sh --sign-script` 或 CMake 变量 `CUSTOM_SIGN_SCRIPT` 指定，CMake 层直接调用该脚本。
- **`.py` 脚本**：CMake 层固定调用 `add_header_sign.py` 作为编排器。如需替换签名执行器（默认为 `community_sign_build.py`），通过 CMake 变量 `CUSTOM_SIGN_SCRIPT` 或 `add_header_sign.py` 的 `--sign_script` 参数指定。

| 脚本类型 | 自定义方式 | 调用方式 |
|----------|-----------|----------|
| `.py` | `CUSTOM_SIGN_SCRIPT` 变量或 `--sign_script` 参数 | `python3 add_header_sign.py <dir> <flag> --bios_check_cfg=<cfg> --sign_script=<自定义脚本> --version=<ver>` |
| `.sh` | `build.sh --sign-script` 或 `CUSTOM_SIGN_SCRIPT` | `bash <自定义脚本> <output_sig> <config> <sign_flag>` |

#### `.py` 自定义签名脚本契约

自定义 `.py` 脚本需与 `community_sign_build.py` 遵循相同的调用契约，包括签名调用与产物类型查询：

| 调用方式 | 说明 |
|----------|------|
| `python3 <script> --crl-dir <dir> <file1> [file2 ...]` | 签名模式，对文件列表制作签名，产出 `<file><ext>` |
| `python3 <script> --print-sign-ext` | 查询签名产物扩展名，stdout 打印（如 `.cms`），退出码 0 |
| `python3 <script> --print-certtype` | 查询证书类型，stdout 打印十六进制（如 `0xFFFFFFFF`），退出码 0 |

编排器在签名前分别调用两个查询 flag 获取产物扩展名与证书类型，用于构造 `image_pack.py` 的 `-cms` 路径和 `-certtype` 参数。查询失败时编排器回退默认值（`.p7s`/`1`），不影响签名流程。

| 证书类型 | 值 | 说明 |
|----------|-----|------|
| 社区证书 | `0x1` | `community_sign_build.py` 使用 |
| 客户端证书 | `0x2` | 自定义脚本可选 |
| 厂商证书 | `0xFFFFFFFF` | 自定义脚本可选 |

示例：使用产出 `.cms` 签名、厂商证书的自定义脚本：

```bash
# 通过 CMake 变量指定（build.sh 透传）
sh build.sh --pkgs=runtime --enable-sign --sign-script /path/to/custom_sign_build.py

# 或直接调用编排器
python3 scripts/sign/add_header_sign.py /path/to/images true \
    --bios_check_cfg=bios_check_cfg.xml \
    --sign_script=/path/to/custom_sign_build.py \
    --version=8.0.0
```

## 配置文件格式

`bios_check_cfg.xml` 声明每个镜像的签名属性：

```xml
<bios_check_cfg>
  <item input="firmware/a.bin"
        output="out"
        type="cms"
        tag="a_firmware"
        version="1.0"
        nvcnt="3"
        position="after_header"/>
  <item input="driver/b.bin"
        output="out"
        type=""
        tag="b_driver"
        version="1.0"/>
</bios_check_cfg>
```

### 属性说明

| 属性 | 说明 | 默认值 | 必选条件 |
|------|------|--------|----------|
| `input` | 镜像文件相对路径 | — | 必选 |
| `output` | 输出标识 | — | 必选 |
| `version` | 版本号 | 由 `--version` 参数传入 | — |
| `type` | 签名类型，`cms` 表示需要 CMS 签名，留空表示不签名 | `""` | — |
| `tag` | 镜像名称标签 | `""` | `type` 含 `cms` 时必选 |
| `nvcnt` | 安全版本号，配置后触发 ESBC 头绑定 | `""` | — |
| `position` | 头与镜像的相对位置：`before_header`/`after_header` | `""` | — |
| `sign_alg` | 签名算法 | `PKCSv1.5` | — |
| `additional` | 传递给 `image_pack.py` 的额外参数 | `""` | — |
| `image_pack` | 镜像打包版本 | `1.0` | — |
| `fw_version` | 固件版本号 | `""` | — |
| `rootrsa` | RSA 根证书标识 | `default_rsa_rootkey` | — |
| `subrsa` | RSA 子证书标识 | `default_rsa_subkey` | — |
| `bist_flag` | BIST 标志 | `""` | — |

## 命令行参数

### add_header_sign.py

```
python3 add_header_sign.py <sign_file_dir> [sign_flag] [options]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `sign_file_dir` | 待签名文件的根目录（必选） | — |
| `sign_flag` | 是否签名：`true`/`false` | `false` |
| `--bios_check_cfg` | 签名配置 XML 文件路径 | `bios_check_cfg.xml` |
| `--version` | 版本号 | — |
| `--sign_script` | 签名插件脚本路径 | 未设置时使用 `scripts/sign/community_sign_build.py` |

返回码：`0` 成功，`1` 失败。

### community_sign_build.py

```
python3 community_sign_build.py [options] <file1> [file2 ...]
```

| 参数 | 说明 |
|------|------|
| `files` | 待签名文件列表（签名模式必选，查询模式不需要） |
| `--crl-dir` | CRL 输出目录，未指定时下载到脚本目录 |
| `--print-sign-ext` | 打印签名产物扩展名（`.p7s`）并退出，不执行签名 |
| `--print-certtype` | 打印证书类型（`0x1`）并退出，不执行签名 |

返回码：`0` 成功，`1` 失败。

## 示例工作流

### 不签名模式（社区开发者常用）

仅加 ESBC 头和 BIOS 字节头，不制作 CMS 签名：

```bash
# 1. 准备镜像文件
mkdir -p /tmp/images/firmware
cp my_firmware.bin /tmp/images/firmware/

# 2. 编写配置文件
cat > /tmp/bios_check_cfg.xml << 'EOF'
<bios_check_cfg>
  <item input="firmware/my_firmware.bin"
        output="out"
        type=""
        tag="my_firmware"
        version="1.0"
        nvcnt="1"/>
</bios_check_cfg>
EOF

# 3. 执行加头（不签名）
python3 scripts/sign/add_header_sign.py /tmp/images false \
    --bios_check_cfg=/tmp/bios_check_cfg.xml \
    --version=1.0
```

### 签名模式（CI/CD 环境）

完整签名流程：ESBC 头 → ini 摘要 → CMS 签名 → 绑定文件头：

```bash
# 1. 确保 signatrust_client 可用
export SIGN_CLIENT_PATH=/path/to/signatrust_client
export SIGN_CLIENT_CONFIG=/path/to/client.toml

# 2. 通过 build.sh 启用签名
sh build.sh --pkgs=runtime --enable-sign

# 或直接调用
python3 scripts/sign/add_header_sign.py /tmp/images true \
    --bios_check_cfg=/tmp/bios_check_cfg.xml \
    --version=8.0.0 \
    --sign_script=scripts/sign/community_sign_build.py
```

### 使用本地 CRL 文件

避免每次签名都从远程下载 CRL：

```bash
# 预下载 CRL 到本地
export CRL_FILE_PATH=/path/to/SWSCRL.crl

# 正常执行签名
python3 scripts/sign/add_header_sign.py /tmp/images true \
    --bios_check_cfg=/tmp/bios_check_cfg.xml \
    --version=8.0.0
```

## 常见问题

**Q: 签名失败提示 "command not found: signatrust_client"**

A: signatrust_client 未安装或路径不正确。设置 `SIGN_CLIENT_PATH` 环境变量指向正确路径，或使用不签名模式（`sign_flag=false`）。

**Q: CRL 下载失败**

A: 设置 `CRL_FILE_PATH` 环境变量指向本地已有的 CRL 文件，避免远程下载。或检查网络连通性和 `CRL_DOWNLOAD_URL` 是否可访问。

**Q: 提示 "bios esbc tool dir not exists"**

A: `scripts/signtool/esbc_header/` 目录不存在。确保 cmake 仓库完整检出。

**Q: 镜像已有头，提示 "No need to add head again"**

A: `esbc_header.py` 和 `image_pack.py` 会检测镜像头的 magic number（`0x55aa55aa`），已加头的镜像会被跳过。如需重新加头，先用 `ci_img_headler.py --rcvr` 提取原始镜像。

**Q: `sign_flag` 的三种行为分别是什么？**

A: `sign_flag` 控制签名流程的执行范围，结合配置文件的 `type` 属性共三种行为：

| `sign_flag` | 配置 `type` | 执行内容 | 产出 |
|-------------|------------|----------|------|
| `"false"` | 任意 | 完全跳过，不执行任何操作 | 无 |
| `"true"` | `""`（空） | 加 ESBC 头 + BIOS 基础头 | 含头的镜像（无签名） |
| `"true"` | `"cms"` | 加 ESBC 头 + ini 摘要 + CMS 签名 + BIOS 完整头 | 含头+签名的镜像 |

通过 `build.sh` 调用时，`ENABLE_SIGN=OFF` 对应 `sign_flag=false`（完全跳过），`ENABLE_SIGN=ON` 对应 `sign_flag=true`（按配置文件的 `type` 决定是否签名）。

## 并行构建支持

> **签名流程支持并行执行。** `make -jN` 下多个签名目标可同时运行，每个目标有独立的中间产物目录。

**隔离机制**：CMake 层为每个签名目标创建独立的 `signatures_${safe_input_name}/` 目录；Python 脚本层的所有中间产物（`sign_tmp/`、`SWSCRL.crl`、`SWSCRL.der`）均在此目录下，无任何路径在源码树下共享。

**CRL 建议**：设置 `CRL_FILE_PATH` 环境变量指向本地 CRL 可避免每目标重复下载（CRL 仅几 KB，未设时重复下载开销可忽略）。

**隔离原理**：参见 [sign-internals.md](sign-internals.md) §7 并行签名设计。

## 测试

```bash
# 运行全部签名测试
python3 -m pytest scripts/sign/tests/ -v

# 运行单个测试文件
python3 -m pytest scripts/sign/tests/test_add_header_sign.py -v
python3 -m pytest scripts/sign/tests/test_community_sign_build.py -v
```

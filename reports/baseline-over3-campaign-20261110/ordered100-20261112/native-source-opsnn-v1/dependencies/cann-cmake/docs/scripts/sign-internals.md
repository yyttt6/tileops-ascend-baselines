<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# 代码签名实现说明

> 脚本位置：`scripts/sign/`（`add_header_sign.py`、`community_sign_build.py`）及 `scripts/signtool/`（`esbc_header/`、`image_pack/`、`image_extract/`）

本文档面向维护者，介绍签名系统的设计原理、模块职责、数据流和关键设计决策。使用指南参见 [sign-guide.md](sign-guide.md)。

## 1. 设计目标

CANN 镜像在发布前需要完成安全加固：绑定版本信息、安全版本号、镜像摘要，并在需要时制作 CMS 数字签名。签名系统需要满足以下目标：

- **配置驱动**：不同镜像有不同的签名需求（是否签名、签名类型、版本号、安全版本号等），通过 XML 配置文件声明，而非硬编码。
- **流程编排**：签名涉及多个步骤（加 ESBC 头、生成摘要、CMS 签名、绑定文件头），需要一个编排器协调各步骤的执行顺序和错误处理。
- **可替换签名后端**：社区版使用 signatrust_client 制作 CMS 签名，企业内部可能使用不同的签名服务，签名后端应可替换。
- **安全执行**：子进程调用需防止命令注入；临时文件需可靠清理，防止残留被打包。
- **CMake 集成**：签名步骤应作为 CMake 构建流程的一部分，在 `build.sh --enable-sign` 时自动触发。

## 2. 模块架构

### 2.1 角色分层

签名系统分为三层，各层职责清晰、单向依赖：

```
┌─────────────────────────────────────────────────────────────────┐
│  CMake 集成层                                                     │
│  add_cann_sign_file() — 创建自定义命令，拷贝文件到签名目录后触发签名 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 调用
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  编排层（scripts/sign/）                                          │
│  add_header_sign.py — 解析配置、调度 4 步流程、错误处理、临时清理   │
│  community_sign_build.py — CMS 签名执行（CRL 准备、签名、校验）    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 调用
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  工具层（scripts/signtool/）                                      │
│  esbc_header.py — 写入 256 字节 ESBC 二级头                       │
│  ini_gen.py — 计算镜像 SHA256 摘要                                │
│  image_pack.py — 绑定完整镜像头（0x600 字节）                     │
│  ci_img_headler.py — 从已加头镜像提取原始镜像                      │
└─────────────────────────────────────────────────────────────────┘
```

- **CMake 集成层**只负责"何时签名"和"签什么文件"，不关心签名细节。
- **编排层**负责"怎么签名"——解析配置、按序调度工具、处理错误和清理。
- **工具层**是原子操作，每个工具完成一个独立的镜像处理任务，被编排层通过子进程调用。

### 2.2 模块职责

| 模块 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `add_header_sign.py` | 编排器：解析配置、调度 4 步流程 | 镜像目录 + 配置 XML + 签名标志 | 加头后的镜像 |
| `community_sign_build.py` | CMS 签名执行器 | 待签名文件列表 | `.p7s` 签名文件 |
| `esbc_header.py` | ESBC 头工具 | 原始镜像 + 版本/nvcnt/tag | 加了 256B 头的镜像 |
| `ini_gen.py` | 摘要生成工具 | 镜像列表 XML | 每个镜像的 `.ini` 摘要文件 |
| `image_pack.py` | 镜像加头工具 | 镜像 + 签名/摘要/CRL | 加了完整头的镜像 |
| `ci_img_headler.py` | 镜像提取工具 | 已加头镜像 | 原始镜像（逆向操作） |

### 2.3 调用链

```
build.sh --enable-sign
    │
    ▼
add_cann_sign_file (CMake)
    │  拷贝 INPUT → signatures_${safe_input_name}/ 目录（每目标独立）
    │  按 ENABLE_SIGN 设置 sign_flag
    │  按脚本扩展名分发（.py / .sh）
    │
    ▼
add_header_sign.py (编排器)
    │
    ├── 解析 bios_check_cfg.xml → 配置字典
    │
    └── add_bios_header (4 步流程，开始前清理残留 sign_tmp)
            │
            ├── 步骤1: 加 ESBC 头 → esbc_header.py
            ├── 步骤2: 生成 ini 摘要 → ini_gen.py
            ├── 步骤3: CMS 签名 → community_sign_build.py → signatrust_client
            └── 步骤4: 绑定文件头 → image_pack.py
```

## 3. 编排器设计原理

### 3.1 四步流程

`add_header_sign.py` 的核心是 `add_bios_header` 函数，按固定顺序编排四个步骤：

| 步骤 | 函数 | 作用 | 执行条件 |
|------|------|------|----------|
| 1. 加 ESBC 头 | `add_bios_esbc_header` | 对配置了 nvcnt 的镜像写入 256 字节 ESBC 二级头 | 总是执行（按 nvcnt 逐镜像判断） |
| 2. 生成 ini 摘要 | `build_inifile` | 为 CMS 类型镜像生成 SHA256 摘要文件 | 仅签名模式 |
| 3. CMS 签名 | `build_sign` | 对所有 CMS 类型镜像的 `.ini` 摘要制作 p7s 签名 | 仅签名模式 |
| 4. 绑定文件头 | `build_image_pack_cmd` + `image_pack.py` | 对每个镜像绑定最终文件头 | 总是执行（命令按签名模式区分） |

**步骤顺序的原理**：

- ESBC 头必须先加，因为它包含镜像的 SHA256 摘要，而后续步骤会修改镜像内容。
- ini 摘要在签名前生成，因为签名工具签的是 ini 文件（内含镜像哈希），而非直接签镜像。
- CMS 签名在绑头前完成，因为绑头时需要把 p7s 签名文件嵌入镜像尾部。
- 绑头是最后一步，它将所有产物（签名、摘要、CRL）打包进最终镜像。

### 3.2 两种运行模式

| 模式 | sign_flag | 执行步骤 | 产出 |
|------|-----------|----------|------|
| 不签名模式 | `"true"` | 步骤 1 + 步骤 4（基础头） | 含 ESBC 头 + BIOS 字节头的镜像 |
| 签名模式 | `"true"` | 步骤 1-4（完整流程） | 含 ESBC 头 + BIOS 头 + CMS 签名 + ini 摘要 + CRL 的镜像 |
| 跳过模式 | `"false"` | 无（直接返回） | 无操作 |

**`sign_flag == "false"` 的早退设计**：当 `ENABLE_SIGN=OFF` 时，CMake 层将 `sign_flag` 设为 `"false"`，编排器直接返回成功，不执行任何加头或签名操作。这意味着"不签名"和"完全不处理"是同一种状态。如果需要"加头但不签名"，应使用 `sign_flag="true"` 并在配置文件中将 `type` 留空。

### 3.3 配置解析设计

配置文件 `bios_check_cfg.xml` 中每个 `<item>` 节点描述一个镜像的签名属性。解析流程：

1. **XML 读取**：使用 `xml.etree.ElementTree` 解析，解析失败返回 `None` 而非抛异常。
2. **属性校验**：`input` 和 `output` 为必选属性；`type` 含 `cms` 时 `tag` 必选。
3. **版本填充**：节点未配置 `version` 时用命令行传入的 `--version` 参数补齐。
4. **文件过滤**：检查镜像文件是否存在，不存在的文件 warning 后跳过，不中断流程。
5. **配置对象构造**：通过 `AddHeaderConfig.from_xml()` 工厂方法解析属性，未配置的属性使用默认值。

**`AddHeaderConfig` 的 dataclass 设计**：

配置类使用 `@dataclass` 装饰器，配合一个 `_CONFIG_ATTR_MAP` 映射表（XML 属性名 → 字段名 → 默认值的三元组列表），`from_xml()` 遍历此表解析属性。这种方式避免了为每个属性重复编写 `if attr in node.attrib` 模式，新增属性只需在映射表和 dataclass 中各加一行。

### 3.4 临时文件管理

签名流程会产生中间文件（镜像副本、ini 摘要、p7s 签名），存放在 `sign_tmp/` 临时目录下。

**设计原则**：
- 临时目录创建在 `sign_file_dir` 下（`sign_tmp/`），复用 CMake 层每目标唯一的 staging 目录实现并行隔离（详见 §7 并行签名设计）。
- CRL/DER 等中间产物同样放在 `sign_file_dir` 下，无任何路径在源码树下共享。
- 流程开始前删除上次残留的 `sign_tmp/` 目录，避免残留文件干扰本次签名；流程结束后保留 `sign_tmp/` 目录以便定位问题。
- 残留目录的最终清理需全量清理构建目录（`rm -rf build/`）；`make clean` 不删除 staging 目录（详见 §7.4）。

**目录结构保持**：临时目录内的文件保持与原始镜像相同的相对目录结构，避免不同子目录下的同名文件冲突。所有 `os.makedirs` 调用使用 `exist_ok=True`，允许同一目录被多次创建。

### 3.5 一次性签名设计

`build_sign` 采用两阶段设计：

- **第一阶段（收集+拷贝）**：遍历配置，将所有需要 CMS 签名的文件拷贝到临时目录，收集对应的 ini 文件路径。
- **第二阶段（一次性签名）**：将所有 ini 文件路径作为参数，一次性调用 `community_sign_build.py`。

**为什么不逐文件签名**：

| 方案 | 子进程调用次数 | CRL 准备次数 | 风险 |
|------|---------------|-------------|------|
| 逐文件调用 | N | N | 旧实现中命令在循环中累积，导致重复签名 |
| 一次性调用 | 1 | 1 | 无 |

`community_sign_build.py` 内部也是逐文件签名，但 CRL 只准备一次。一次性调用减少了子进程启动开销，且避免了命令累积 bug。

### 3.6 ini 摘要文件格式

`ini_gen.py` 为每个 `type="cms"` 的镜像生成一个 `.ini` 摘要文件，内容为**单行文本**，格式为 `tag,   SHA256哈希值;`：

```
a_firmware,   e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855;
```

| 字段 | 说明 | 来源 |
|------|------|------|
| `tag` | 镜像标签，对应配置文件中的 `tag` 属性 | `bios_check_cfg.xml` 的 `<item tag="...">` |
| `SHA256 哈希值` | 镜像文件的 SHA256 摘要（64 位十六进制字符串） | `ini_gen.py` 读取镜像文件计算（`hashlib.sha256`，4K 分块） |

**生成逻辑**（`ini_gen.py:85-102`）：
1. 读取 `image_info.xml` 中每个 `<image>` 节点的 `path`（镜像路径）和 `tag`
2. 用 SHA256 算法计算镜像文件的哈希值（`cal_image_hash`，4K 分块读取）
3. 将 `tag` 和哈希值用 `,   `（逗号+3空格）拼接，末尾加 `;\n`，写入 `${ini_name}.ini`

**为什么用这种格式**：这是 `image_pack.py` 绑头时解析 ini 的约定格式。绑头时 `image_pack.py` 通过 `-ini` 参数读取该文件，提取 tag 和哈希值嵌入 BIOS 头的 `code_hash` 字段（64 字节）。验证方启动时重新计算镜像 SHA256，与 ini 中的哈希比对，验证镜像完整性。

> 注意：`position="before_header"` 的镜像使用 `cal_fs_image_hash`，哈希值后追加 `;dm-roothash,<roothash>`，用于 dm-verity 文件系统镜像的根哈希校验。普通镜像使用 `cal_image_hash`，仅含纯 SHA256 哈希。

### 3.7 产物类型查询机制

编排器在签名前通过两个查询 flag 获取签名脚本的产物扩展名与证书类型，用于构造 `image_pack.py` 命令：

| 查询 flag | 返回值 | 示例 |
|-----------|--------|------|
| `--print-sign-ext` | 签名产物扩展名（含点号） | `.p7s` / `.cms` |
| `--print-certtype` | 证书类型（十六进制字符串） | `0x1` / `0x2` / `0xFFFFFFFF` |

**设计原理**：签名格式与证书类型是签名执行器的固有属性，而非外部配置。让脚本自己声明（`print` 后退出），编排器被动接收，消除了"外部配置与脚本实际行为不一致"的配置漂移风险。

**查询流程**（`add_bios_header` 步骤3前）：

```
query_sign_ext(sign_tool_path)
  → python3 <sign_script> --print-sign-ext
  → stdout 最后一行非空行 = ".p7s"
  → 失败时回退 ".p7s"

query_certtype(sign_tool_path)
  → python3 <sign_script> --print-certtype
  → stdout 最后一行非空行 = "0x1"
  → str(int("0x1", 0)) = "1"  # 转十进制
  → 失败时回退 "1"
```

**向后兼容**：旧脚本不识别查询 flag 时，argparse 报错退出，`safe_run_cmd` 返回失败，编排器回退默认值（`.p7s`/`1`），签名正常进行。这是因为查询是独立的预调用，失败不影响后续签名命令。

**并行安全**：查询是无状态操作，脚本只 `print` 静态常量并退出，不读写文件、不依赖全局状态，多个签名目标并行查询无竞争。

## 4. CMS 签名执行器设计原理

### 4.1 签名流程

`community_sign_build.py` 负责实际的 CMS 签名制作，流程分三步：

1. **准备 CRL**：CRL（证书吊销列表）用于验证签名证书的有效性。
2. **逐文件签名**：调用 signatrust_client 对每个 `.ini` 摘要文件制作 detached p7s 签名（签名对象是 ini 摘要，不是镜像本身）。
3. **校验结果**：检查每个 `.ini.p7s` 文件是否存在且非空。

### 4.2 CRL 准备策略

```
CRL_FILE_PATH 已设置且文件存在？
    ├── 是 → 直接使用本地文件（跳过下载）
    └── 否 → 从远程下载到脚本目录
            ├── 最多重试 3 次（含超时处理）
            └── 全部失败 → 返回 None，中止签名
```

**设计要点**：
- **本地优先**：`CRL_FILE_PATH` 环境变量允许用户指定本地 CRL 文件，避免每次签名都远程下载。这对离线环境和网络受限环境很重要。
- **下载到每目标独立目录**：CRL 下载到 `sign_file_dir/`（由 `--crl-dir` 参数传入），每目标隔离避免并行冲突。`CRL_FILE_PATH` 环境变量可指定全局共享的本地 CRL，跳过下载。
- **重试机制**：网络下载可能失败，最多重试 3 次，每次有 60 秒超时。`check=False` 让 curl 失败时不抛异常，通过返回码判断，统一在循环中处理。

### 4.3 签名命令构建

signatrust_client 的调用参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--config` | `SIGN_CLIENT_CONFIG` | signatrust_client 配置文件 |
| `add` | — | 子命令：添加签名 |
| `--file-type` | `p7s` | 输出格式：PKCS#7 detached 签名 |
| `--key-type` | `x509` | 证书类型：x509 |
| `--key-name` | `SignCert` | 签名证书名称 |
| `--detached` | — | detached 模式（签名与文件分离） |
| `--timestamp-key` | `TimeCert` | 时间戳证书名称 |
| `--crl` | CRL 路径 | 证书吊销列表 |

命令以 list 格式构建，配合 `shell=False` 执行，消除命令注入风险。

### 4.4 签名校验

签名完成后，对每个已签名文件校验其对应的 `.p7s` 文件：

- **存在性校验**：p7s 文件必须存在。
- **非空校验**：p7s 文件大小必须大于 0，防止残留空文件误判成功。
- **空列表守卫**：如果没有任何文件被签名（全部被跳过），视为失败。

p7s 文件由 signatrust_client 生成在输入文件同目录，命名为 `输入文件名 + ".p7s"`。

## 5. 镜像头结构与最终文件格式

### 5.1 两种模式的最终文件格式对比

加头后的镜像文件由多个区域组成，签名模式与不签名模式的区别在于是否包含 CMS 签名区。以 `position="after_header"`（默认）为例：

**不签名模式（加头不签名）**：

```
┌─────────────────────────────────────────┐
│ ESBC 头 (256 字节)                       │  ← esbc_header.py 写入
│   code_tag / nvcnt / code_hash /        │
│   magic(0x3a3aaa33) / version ...       │
├─────────────────────────────────────────┤
│ BIOS 头 (0x600 字节)                     │  ← image_pack.py 写入
│   preamble(0x55AA55AA) / head_len /     │
│   code_hash / RSA 公钥区 / RSA 签名区 /  │
│   magic_number / nvcnt ...              │
├─────────────────────────────────────────┤
│ 原始镜像数据                             │
│   (从偏移 0x2000 开始)                    │
└─────────────────────────────────────────┘
```

**签名模式（加头+签名）**：

```
┌─────────────────────────────────────────┐
│ ESBC 头 (256 字节)                       │  ← esbc_header.py 写入
│   code_tag / nvcnt / code_hash /        │
│   magic(0x3a3aaa33) / version ...       │
├─────────────────────────────────────────┤
│ BIOS 头 (0x600 字节)                     │  ← image_pack.py 写入
│   preamble(0x55AA55AA) / head_len /     │
│   code_hash / RSA 公钥区 / RSA 签名区 /  │
│   magic_number / nvcnt ...              │
├─────────────────────────────────────────┤
│ 原始镜像数据                             │
│   (从偏移 0x2000 开始)                    │
├─────────────────────────────────────────┤
│ CMS 签名区 (4K 对齐)                     │  ← 仅签名模式
│   12B 头标识("cms"+长度) + p7s 签名数据  │
├─────────────────────────────────────────┤
│ ini 摘要区 (16K 对齐)                    │  ← 仅签名模式
│   12B 头标识("ini"+长度) + SHA256 摘要   │
├─────────────────────────────────────────┤
│ CRL 区 (2K 对齐)                         │  ← 仅签名模式
│   12B 头标识("crl"+长度) + CRL 吊销列表  │
└─────────────────────────────────────────┘
```

**两种模式的差异**：

| 区域 | 不签名模式 | 签名模式 |
|------|-----------|----------|
| ESBC 头 (256B) | 有（nvcnt 非空时） | 有（nvcnt 非空时） |
| BIOS 头 (0x600B) | 有（基础参数：version/nvcnt/tag） | 有（完整参数：含 cms/ini/crl 路径） |
| 原始镜像数据 | 有 | 有 |
| CMS 签名区 | 无 | 有（p7s detached 签名） |
| ini 摘要区 | 无 | 有（SHA256 哈希值） |
| CRL 区 | 无 | 有（证书吊销列表 DER 格式） |

签名模式比不签名模式在文件尾部多出三个追加区（CMS 签名 + ini 摘要 + CRL），每个区前有 12 字节的头标识（`"cms"`/`"ini"`/`"crl"` + 4 字节长度），便于提取工具按类型定位和解析。

### 5.2 ESBC 二级头字段布局（256 字节）

由 `esbc_header.py` 写入，包含镜像的安全元数据：

| 字段 | 大小 | 说明 |
|------|------|------|
| code_tag | 16B | 镜像标签（ASCII） |
| nvcnt | 4B | 安全版本号（用于防回滚） |
| hash_alg | 4B | 哈希算法（0 = SHA256） |
| code_hash | 64B | 镜像 SHA256 摘要 |
| code_offset | 4B | 镜像数据偏移（before_header=0, after_header=0x100） |
| code_len | 4B | 镜像数据长度 |
| ver_value | 16B | 版本字符串 |
| magic_num | 4B | ESBC magic（`0x3a3aaa33`） |
| sign_enable_field | 4B | 签名使能标志（`0x4`） |
| hashtree_offset | 4B | 哈希树偏移（`0x20000` = 128K） |
| hw_logic_version | 4B | 硬件逻辑版本 |
| image_version | 4B | 镜像版本（magic `0x564D` + version 0） |
| hwheader_offset | 4B | 硬件头偏移（256） |
| reserved | 112B | `0xff` 填充 |

**position 语义**：
- `after_header`（默认）：ESBC 头在镜像数据前面（偏移 0x100），头在前数据在后。
- `before_header`：ESBC 头在镜像数据后面，数据在前头在后。

**幂等保护**：写入前检测镜像头的 magic number（`0x55aa55aa`），已加头的镜像跳过，避免重复加头。

### 5.3 BIOS 完整头字段布局（0x600 字节）

由 `image_pack.py` 委托 `hi_platform/platform.py` 写入，包含完整的镜像描述信息：

```
偏移 0x0000  preamble (0x55AA55AA)          — 头标识
偏移 0x0004  head_len (0x600)               — 头长度
偏移 0x0030  code_hash (32B SHA256)          — 镜像哈希
偏移 0x0600  RSA 公钥区 (N + E 各 512B)      — RSA 证书
偏移 0x0E00  RSA 签名区 (512B)               — RSA 签名
偏移 0x2000  原始镜像数据                    — 镜像内容
偏移 0x0580  magic_number + file_size        — 文件标识和大小
偏移 0x0590  nvcnt_magic(0x5A5AA5A5) + nvcnt — 安全版本号
文件尾部     CMS 签名(4K对齐) + ini(16K对齐) + CRL(2K对齐)
```

**尾部追加设计**：CMS 签名、ini 摘要、CRL 按各自的对齐要求追加在文件尾部，每个区块前有 12 字节的头标识（`"cms"`/`"ini"`/`"crl"` + 长度），便于提取工具定位。

### 5.4 certtype 证书类型

`image_pack.py` 的 `-certtype` 参数区分签名证书来源：

| 值 | 含义 |
|-----|------|
| `0x1` | 社区证书（Community Certificate）— 社区版签名使用 |
| `0x2` | 客户端证书（Client Certificate） |
| `0xFFFFFFFF` | 厂商证书（HW Certificate）— `image_pack.py` 默认值 |

编排器通过查询签名脚本（`--print-certtype` flag）获取证书类型，不再硬编码。`community_sign_build.py` 声明 `0x1`；自定义签名脚本可声明 `0x2` 或 `0xFFFFFFFF`。查询失败时回退 `0x1`（兼容未更新的第三方脚本）。

> **注意**：`image_pack.py` 的 `-certtype` 使用 `type=int`，编排器需将十六进制（如 `0xFFFFFFFF`）转为十进制字符串（`4294967295`）后传入，否则 `int("0xFFFFFFFF")` 报错。

## 6. CMake 集成原理

### 6.1 add_cann_sign_file 函数

CMake 公共 API `add_cann_sign_file` 创建一个自定义命令和目标：

1. **输入归一化**：相对路径的 `INPUT` 解析为 `${CMAKE_CURRENT_BINARY_DIR}` 下的绝对路径。
2. **输出路径**：自动生成 `${CMAKE_CURRENT_BINARY_DIR}/signatures_${safe_input_name}/<input_name>`（每目标独立目录）。
3. **脚本选择**：`CUSTOM_SIGN_SCRIPT` 变量覆盖默认签名脚本，默认为 `community_sign_build.py`；`.py` 和 `.sh` 分支均使用 `${SIGN_SCRIPT}`。
4. **签名标志**：`ENABLE_SIGN` 变量转换为 `"true"`/`"false"` 字符串。
5. **脚本分发**：按脚本扩展名分发——`.py` 走完整编排流程，`.sh` 直接执行。
6. **自定义命令**：`make_directory → copy → sign_cmd` 三步，`VERBATIM` 保证参数安全传递。
7. **返回路径**：签名文件路径通过 `RESULT_VAR` 返回给调用方。

### 6.2 脚本分发机制

| 脚本类型 | 调用方式 | 适用场景 |
|----------|----------|----------|
| `.py` | `python3 add_header_sign.py <dir> <flag> --bios_check_cfg=<cfg> --sign_script=<builder> --version=<ver>` | 社区版默认流程 |
| `.sh` | `bash <script> <output> <config> <flag>` | 企业自定义流程 |

`.py` 脚本走完整的 4 步编排流程（`add_header_sign.py` → `community_sign_build.py`）。`.sh` 脚本直接执行，由脚本自行处理全部逻辑。

### 6.3 关键变量流转

```
build.sh --enable-sign
    → CMake: ENABLE_SIGN=ON
        → sign_flag="true"
            → add_header_sign.py: add_sign="true"
                → build_inifile/build_sign 执行
                → build_image_pack_cmd 构建完整命令
```

版本号经 CMake 的 `--version` 参数传递到编排器，最终写入镜像头的 `ver_value` 字段。版本号由 `add_cann_sign_file` 的 `VERSION` 参数（必选）提供。

## 7. 并行签名设计

签名流程支持 `make -jN` 并行构建，多个签名目标可同时运行。本节说明隔离原理和路径布局。

### 7.1 隔离原理

并行安全的核心是**每目标独立的中间产物路径**。CMake 层为每个签名目标创建独立的 staging 目录，Python 脚本层的所有中间产物均在此目录下，无任何路径在源码树下共享：

```
${CMAKE_CURRENT_BINARY_DIR}/signatures_${safe_input_name}/    ← CMake 层创建，每目标唯一
    ├── ${input_name}                                          ← 最终输出（image_pack.py 原地覆写加头后的镜像）
    ├── SWSCRL.crl                                             ← CRL 下载（未设 CRL_FILE_PATH 时）
    ├── SWSCRL.der                                             ← DER 转换产出
    └── sign_tmp/                                              ← 临时目录（开始前清理残留，结束后保留）
        ├── image_info.xml                                     ← ini_gen.py 输入（镜像清单）
        ├── ${ini_name}.ini                                    ← ini_gen.py 产出（镜像 SHA256 摘要，签名对象）
        ├── ${file}                                            ← 拷贝的镜像副本（供签名工具引用，非签名对象）
        └── ${ini_name}.ini.p7s                                ← signatrust_client 产出（对 .ini 的 CMS detached 签名）
```

> **签名对象说明**：CMS 签名针对的是 `${ini_name}.ini`（镜像的 SHA256 摘要文件），而非镜像本身。`${file}` 是镜像副本，供签名工具引用路径，但不是被签名的对象。绑头时 `image_pack.py` 将 p7s 签名和 ini 摘要同时嵌入镜像头，验证方既验签名合法性又比对镜像哈希完整性。

### 7.2 两层隔离

| 层 | 路径来源 | 隔离机制 |
|---|---------|---------|
| **CMake 层** | `signatures_${safe_input_name}/` | `function/prepare.cmake` 按输入文件名生成目录名，每目标唯一 |
| **Python 层** | `sign_file_dir/sign_tmp/`、`sign_file_dir/SWSCRL.crl`、`sign_file_dir/SWSCRL.der` | `add_header_sign.py` 和 `community_sign_build.py` 基于 `sign_file_dir`（即 CMake 层传入的 staging 目录）推导所有中间产物路径 |

`sign_file_dir` 作为第一个位置参数从 CMake 层传入 `add_header_sign.py`，再通过 `--crl-dir` 传入 `community_sign_build.py`，确保两层使用同一隔离目录。

### 7.3 CRL 复用与隔离

CRL 文件路径的解析顺序（`add_header_sign.py` 与 `community_sign_build.py` 一致）：

1. `CRL_FILE_PATH` 环境变量已设且文件存在 → 直接复用（CI 环境预置，不触发下载）
2. 否则 → 下载/生成到 `sign_file_dir/SWSCRL.crl`（每目标独立）

CI 环境设置 `CRL_FILE_PATH` 可避免每目标重复下载；未设时 CRL 仅几 KB，每目标各下载一份的开销可忽略。DER 由 CRL 转换而来，同样放在 `sign_file_dir` 下，每目标独立目录消除了跨进程 TOCTOU 竞争。

### 7.4 清理策略

- `sign_tmp/` 子目录在每次 `add_bios_header` 开始前删除上次残留，流程结束后保留以便定位问题。
- staging 目录 `signatures_${safe_input_name}/` 及其中的 CRL/DER 未注册为 CMake 的清理目标，`make clean` 不会删除它们；这些文件持续存在直到全量清理构建目录（`rm -rf build/`）。

## 8. 安全设计

### 8.1 命令执行安全

所有子进程调用统一通过 `safe_run_cmd` 函数：

- **`shell=False` + list 参数**：所有命令以列表形式传递，不经过 shell 解析，消除命令注入风险。
- **`additional` 参数安全拆分**：配置文件中的 `additional` 属性通过 `shlex.split()` 拆分为列表，而非直接拼入命令字符串。
- **`FileNotFoundError` 捕获**：可执行文件不存在时返回失败而非抛异常，保证编排器能正常记录错误并返回失败状态。

### 8.2 临时文件管理

`sign_tmp/` 临时目录在每次签名开始前清理上次残留，结束后保留以便定位问题。临时目录包含：
- 镜像副本（可能含敏感数据）
- ini 摘要文件
- p7s 签名文件

临时目录位于构建目录下的 `signatures_${safe_input_name}/` staging 目录中，不会被打包到发布包。staging 目录未注册为 CMake 清理目标，`make clean` 不删除；需全量清理构建目录（`rm -rf build/`）才能清除。

## 9. 数据流

### 9.1 不签名模式

```
配置文件 + 原始镜像
    │
    ▼
解析配置 → AddHeaderConfig 字典
    │
    ▼
加 ESBC 头（仅 nvcnt 非空的镜像）
    │
    ▼
跳过 ini 摘要和 CMS 签名
    │
    ▼
绑定 BIOS 基础头（version/nvcnt/tag）
    │
    ▼
最终镜像（ESBC 头 + BIOS 基础头）
```

### 9.2 签名模式

```
配置文件 + 原始镜像
    │
    ▼
解析配置 → AddHeaderConfig 字典
    │
    ▼
加 ESBC 头
    │
    ▼
生成 image_info.xml → ini_gen.py → .ini 摘要文件（SHA256）
    │
    ▼
拷贝镜像到临时目录 → community_sign_build.py → signatrust_client
    │                                                  ↓
    │                              对 .ini 摘要（非镜像）做 CMS detached 签名 → .p7s 签名
    │
    ▼
CRL → DER 格式转换
    │
    ▼
绑定 BIOS 完整头（含 p7s 签名 + ini 摘要 + CRL）
    │
    ▼
最终镜像（ESBC 头 + BIOS 完整头 + CMS 签名 + ini 摘要 + CRL）
    │
    ▼
清理临时目录
```

## 10. 测试体系

### 10.1 测试覆盖

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_add_header_sign.py` | 配置解析、命令构建、子进程 mock、错误处理、CRL 转换、ESBC 头、完整编排流程、argparse 参数解析、环境变量设置、main 入口、并行隔离（sign_tmp/der/crl 路径位于 sign_file_dir 下） |
| `test_community_sign_build.py` | 常量校验、签名命令构建、CRL 准备（本地优先/远程下载/重试/超时）、单文件签名、签名主流程、main 入口、`--crl-dir` 参数解析与透传、CRL 输出目录隔离 |

### 10.2 测试策略

- **子进程隔离**：所有子进程调用通过 `mock.patch` 替换为模拟对象，不依赖 signatrust_client、openssl、curl 等外部工具。
- **文件系统隔离**：使用 pytest 的 `tmp_path` fixture 创建临时目录，测试间互不影响。
- **环境变量自动设置**：`autouse` fixture 自动设置 `HI_PYTHON` 环境变量。
- **布尔断言**：所有返回值用 `is True`/`is False` 断言，与脚本的布尔返回值约定一致。

## 11. 扩展指南

### 11.1 新增签名类型

当前只支持 `cms` 类型。如需新增（如 `rsa`）：

1. 编排器的签名收集逻辑中新增类型分组。
2. 实现该类型的签名执行器（类似 `community_sign_build.py`）。
3. 命令构建函数中新增对应的 `image_pack.py` 参数分支。
4. 补充对应的单元测试。

### 11.2 替换签名后端

通过 CMake 变量 `CUSTOM_SIGN_SCRIPT` 或 `add_header_sign.py` 的 `--sign_script` 参数指定自定义脚本：

- **`.py` 脚本**：需遵循与 `community_sign_build.py` 相同的调用契约——签名模式接收 `--crl-dir <dir> <file1> [file2 ...]`；查询模式支持 `--print-sign-ext`/`--print-certtype` flag 声明产物扩展名与证书类型。返回 `0`（成功）或 `1`（失败）退出码。
- **`.sh` 脚本**：需接收 `<output_sig> <config> <sign_flag>` 三个位置参数。

编排器在签名前通过查询 flag 获取产物扩展名与证书类型，替换后端不需要修改编排器代码。查询失败时回退默认值（`.p7s`/`0x1`）。

### 11.3 新增配置属性

1. 在配置映射表 `_CONFIG_ATTR_MAP` 中新增 `(xml_attr, field_name, default)` 三元组。
2. 在 `AddHeaderConfig` dataclass 中新增对应字段。
3. 如属性影响命令构建，更新 `build_image_pack_cmd` 函数。
4. 更新单元测试中的配置解析和命令构建测试。

### 11.4 修改镜像头格式

镜像头格式定义在 `scripts/signtool/image_pack/hi_platform/platform.py` 的头构造函数中。修改 `struct.Struct` 格式字符串和对应字段即可，但需同步修改 `ci_img_headler.py` 的提取逻辑，并确保 `esbc_header.py` 的 ESBC 头结构与 BIOS 头中的相关字段保持一致。

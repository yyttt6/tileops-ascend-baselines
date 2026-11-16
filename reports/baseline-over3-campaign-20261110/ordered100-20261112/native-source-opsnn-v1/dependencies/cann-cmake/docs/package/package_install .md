# 打包与安装说明

> 本指南面向社区开发者与运维人员，手把手带你完成 CANN 组件的**打包**与**安装/卸载**。

## 这份文档能帮你做什么

- 你想把自己编译的 CANN 组件打成可分发的安装包（`.run` / `.rpm` / `.deb`）→ 看 [二、打包操作](#二打包操作)
- 你拿到了一个 CANN `.run` / `.rpm` / `.deb` 包，想安装到机器上 → 看 [三、安装操作](#三安装操作)
- 你想卸载已安装的 CANN 组件 → 看 [四、卸载操作](#四卸载操作)
- 你想了解安装后文件权限为什么是这样 → 看 [五、安装权限规范](#五安装权限规范)
- 安装/打包遇到报错 → 看 [七、常见问题](#七常见问题) 与 [八、排错检查清单](#八排错检查清单)

## 文档约定

| 约定 | 含义 |
|------|------|
| `<包名>` | 需要你替换为实际值的占位符，如 `runtime`、`asc-devkit` |
| `<安装根路径>` | 你指定的安装目录，如 `/usr/local/Ascend` |
| `#` 开头的行 | shell 注释，可直接复制执行整段代码块 |
| **注意** / **提示** | 重要提醒，建议先阅读再操作 |

> **不复述内容**：环境准备、仓库布局、`build.sh` 完整参数表、superbuild 设计原理、依赖解析、host/device 编译流程、关键变量、新增包步骤等，已在仓库文档中详述，请直接参阅：
> - 构建用法与参数：[`docs/superbuild/getting-started.md`](https://gitcode.com/cann/cmake/blob/master/docs/superbuild/getting-started.md)
> - 构建机制与流程：[`docs/superbuild/internals.md`](https://gitcode.com/cann/cmake/blob/master/docs/superbuild/internals.md)
> - 安装权限规范：[`docs/install/permissions.md`](https://gitcode.com/cann/cmake/blob/master/docs/install/permissions.md)

## 适用范围

- 操作系统：Linux（x86_64 / aarch64）
- 安装方式：root 安装或普通用户安装、单用户或多用户共享
- 源码版本：`cann/cmake` 仓库 master 分支

---

## 目录

- [一、背景与前置条件](#一背景与前置条件)
  - [1.1 CANN 与 cmake 仓库的关系](#11-cann-与-cmake-仓库的关系)
  - [1.2 前置条件](#12-前置条件)
- [二、打包操作](#二打包操作)
  - [2.1 打包命令](#21-打包命令)
  - [2.2 包类型与产物格式](#22-包类型与产物格式)
  - [2.3 打包机制（产物如何生成）](#23-打包机制产物如何生成)
  - [2.4 产物命名与输出位置](#24-产物命名与输出位置)
  - [2.5 .run 包内部结构](#25-run-包内部结构)
- [三、安装操作](#三安装操作)
  - [3.1 安装 .run 包](#31-安装-run-包)
  - [3.2 安装 rpm 包](#32-安装-rpm-包)
  - [3.3 安装 deb 包](#33-安装-deb-包)
  - [3.4 安装流程（执行了什么）](#34-安装流程执行了什么)
  - [3.5 多版本共存](#35-多版本共存)
  - [3.6 安装后验证](#36-安装后验证)
- [四、卸载操作](#四卸载操作)
  - [4.1 卸载 .run 安装的内容](#41-卸载-run-安装的内容)
  - [4.2 卸载 rpm / deb 包](#42-卸载-rpm--deb-包)
  - [4.3 卸载流程](#43-卸载流程)
- [五、安装权限规范](#五安装权限规范)
- [六、端到端示例](#六端到端示例)
- [七、常见问题](#七常见问题)
- [八、排错检查清单](#八排错检查清单)
- [九、术语表](#九术语表)
- [附录 A：filelist.csv 字段说明](#附录-afilelistcsv-字段说明)
- [附录 B：可打包的包名](#附录-b可打包的包名)
- [附录 C：参考文档与反馈](#附录-c参考文档与反馈)

---

## 一、背景与前置条件

### 1.1 CANN 与 cmake 仓库的关系

CANN 是面向昇腾 AI 处理器的计算架构生态，由多个独立组件仓组成（runtime、metadef、ge、hcomm 等）。`cann/cmake` 仓库是其中的**公共构建框架**，为所有组件提供统一的编译、打包、安装能力。

```
┌─────────────────────────────────────────────────────────────┐
│                    CANN 组件生态                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ runtime  │  │ metadef  │  │  hcomm   │  │   ...    │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       └─────────────┴──────┬──────┴─────────────┘           │
│                            ▼                                │
│                   ┌─────────────────┐                       │
│                   │   cmake（本仓库）│  ← 公共打包/安装框架   │
│                   └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

**关键点**：`cmake` 仓库本身**不产出可运行制品**，也**不能独立打包**——它必须与 CANN 组件仓置于同一父目录下，由 `build.sh` 驱动联合编译并打包，所以"打包"实质是"联合编译 + 收集产物 + 打成安装包"一条龙。

### 1.2 前置条件

打包前请先完成构建环境准备（仓库布局、CANN 工具包、CMake/GCC/Python 等），详见
[`docs/superbuild/getting-started.md`](https://gitcode.com/cann/cmake/blob/master/docs/superbuild/getting-started.md) 的"环境准备"章节。

与打包/安装直接相关的三点：

1. **本仓库不能独立打包**：`cmake` 仓库必须与 CANN 组件仓（runtime、metadef、ge、hcomm 等）置于同一父目录 `CANN_TOP_DIR` 下，由 `build.sh` 驱动 `superbuild/` 联合编译并打包。
2. **device 交叉编译可选**：若仅需 host 侧产物，加 `--build_host_only` 跳过 device 编译，此时无需安装 CANN 工具包（不涉及 hcc 交叉编译工具链）。
3. **安装阶段无需源码**：拿到 `.run`/`.rpm`/`.deb` 产物后，目标机器只需是 Linux，不需要 cmake 仓库或编译工具链。

---

## 二、打包操作

### 2.1 打包命令

打包入口为仓库根目录的 `build.sh`，它依次执行三步：

```bash
cmake -S superbuild -B build      # 1. 配置：解析依赖、生成 Makefile
cmake --build build                # 2. 编译：构建目标包及其依赖
cpack -B build                     # 3. 打包：调用 CPack 生成安装包
```

最简打包（`--pkgs` 必选，默认产出 `.run` 自解压安装包）：

```bash
sh build.sh --pkgs=<包名>
```

按需选择产物格式：

```bash
# 指定包类型
sh build.sh --pkgs=runtime --pkg-type=rpm

# 同时打出 deb 和 rpm
sh build.sh --pkgs=runtime --pkg-type=deb,rpm

# 打出全部类型（run + deb + rpm）
sh build.sh --pkgs=runtime --pkg-type=all

# 多包联合打包，指定线程数与详细输出
sh build.sh --pkgs=runtime,asc-devkit -j16 -v
```

> `--pkgs` 为**必选**参数，多个包名用逗号分隔。可用的包名见 [附录 B](#附录-b可打包的包名)，完整参数表（`-j`、`--build-type`、`--build_host_only`、`--cann_path`、工具链等）见 `getting-started.md`。

与打包产物直接相关的参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pkgs=<PACKAGES>` | **必选**，要打包的组件名，逗号分隔 | — |
| `--pkg-type=<TYPE>` | 产物类型：`run` / `rpm` / `deb` / `deb,rpm` / `all` | `run` |
| `--cann_3rd_lib_path=<PATH>` | 第三方库路径（makeself 等依赖从此查找） | `./output/third_party` |
| `--enable-sign` / `--sign-script <PATH>` | 启用代码签名 / 指定签名脚本 | 关闭 |

### 2.2 包类型与产物格式

由 `set_cann_cpack_config()`（`function/prepare.cmake:494-504`）根据 `--pkg-type` 选择 CPack Generator：

| `--pkg-type` | CPack Generator | 产物 | 适用场景 |
|------|----------------|------|---------|
| `run`（默认） | External（makeself） | `*.run` 自解压安装包 | 通用，跨发行版，内嵌安装脚本，无需包管理器 |
| `rpm` | RPM | `*.rpm` | RPM 系（CentOS / RHEL / openEuler / SUSE） |
| `deb` | DEB | `*.deb` | Debian / Ubuntu 系 |
| `deb,rpm` | DEB;RPM | `*.deb` + `*.rpm` | 同时发布两种系包 |
| `all` | DEB;RPM;External | 三种全部 | 一键产出全部分发格式 |

**我该选哪种？**
- 不确定目标机器发行版、或想内嵌完整安装/卸载脚本 → 选 `run`（默认，最通用）
- 目标机器是 RPM 系、想用 `rpm`/`yum` 管理依赖 → 选 `rpm`
- 目标机器是 Debian 系、想用 `apt` 自动解决依赖 → 选 `deb`
- 要同时发布给两种发行版 → 选 `deb,rpm` 或 `all`

### 2.3 打包机制（产物如何生成）

**`.run` 包**（`scripts/package/makeself.cmake`）：
1. 创建临时 staging 目录 `_CPack_Packages/makeself_staging`
2. 执行 `cmake --install <build> --prefix <staging> --component <component>`，将组件安装到 staging
3. 运行 `pre_package.cmake`：在 staging 内生成 `filelist.csv`（文件清单）、版本信息、`install.sh`/`uninstall.sh` 等
4. 调用 `makeself.sh` 将 staging 目录打包为自解压 `.run`，内嵌 `share/info/<package>/script/install.sh` 作为安装入口（`makeself.cmake:59-67`）
5. 运行 `post_package.cmake`：将 `cann-*.run` 拷贝到 `build_out/`（`post_package.cmake:31-32`）

**`rpm`/`deb` 包**：
- 由 CPack RPM/DEB Generator 生成，安装前缀 `CPACK_PACKAGING_INSTALL_PREFIX=/usr/local/Ascend/cann-<version>`（`prepare.cmake:493`）
- 维护脚本由 `scripts/package/gen_postinst_prerm.py` 生成 `postinst`/`prerm`，并通过 `CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA`、`CPACK_RPM_POST_INSTALL_SCRIPT_FILE`/`CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE` 注入（`prepare.cmake:509-517`），安装/卸载时由 dpkg/rpm 自动调用，内部走与 `.run` 相同的 `install_common_parser.sh` 逻辑
- 包间依赖由各组件 `set_cann_run_dependencies` 声明，自动转换为 `Depends`（deb）/`Requires`（rpm）（`prepare.cmake:474-478`）
- deb 包生成后还会运行 `scripts/package/fix_deb_control.sh` 修正 control 文件末尾空行（`post_package.cmake:11-29`）

**device 侧产物**：device 侧编译后由 `set_cann_subprj_package(SUPERBUILD)` 打成 `device-<pkg>.tar.gz`，再由 host 侧 `install(FILES ...)` 归入对应组件，最终随 host 侧包一起发布（`superbuild/CMakeLists.txt:137-143`）。因此你只需要安装 host 侧包，device 产物已包含在内。

### 2.4 产物命名与输出位置

- 构建目录：`build/`（中间产物）
- 打包产物输出目录：`build_out/`（`prepare.cmake:397-398`，superbuild 模式下 `CPACK_CMAKE_INSTALL_PREFIX=<cmake仓库>/build_out`）

产物命名：

| 产物 | 命名格式 | 示例 |
|------|---------|------|
| `.run` / `.rpm` / `.deb` | `cann-<component>_<version>_<system>-<arch>.<ext>` | `cann-runtime_9.1.0_linux-x86_64.run` |
| 含芯片名 | `cann-<chip>-<component>_<version>_<system>-<arch>.<ext>` | `cann-ascend910-runtime_9.1.0_linux-x86_64.run` |
| device 侧中间产物 | `device-<pkg>.tar.gz` | `device-runtime.tar.gz`（归入 host 包，不单独发布） |

> 命名中的 `<system>` 为小写系统名（如 `linux`），`<arch>` 为 `x86_64` 或 `aarch64`。

### 2.5 .run 包内部结构

`.run` 解压后（staging 目录）的关键内容：

```
<staging>/
├── <arch>-linux/              # 架构相关产物（lib64/、bin/ 等）
├── python/site-packages/      # Python 产物
├── share/info/<package>/      # 包元信息（驱动安装）
│   ├── script/
│   │   ├── install.sh         # 安装入口（makeself 启动脚本）
│   │   ├── uninstall.sh       # 卸载脚本
│   │   └── help.info          # 帮助信息（makeself --help-header）
│   ├── filelist.csv           # 文件清单（驱动建目录/拷文件/设权限，见附录 A）
│   └── ascend_install.info    # 安装信息
└── ...
```

`filelist.csv` 是安装的**核心驱动文件**，安装/卸载脚本据此完成建目录、拷贝/移动、设权限、建软链、删除等动作。它的字段说明见 [附录 A](#附录-afilelistcsv-字段说明)。

---

## 三、安装操作

### 3.1 安装 .run 包

`.run` 是 makeself 自解压包，直接执行即可安装。安装脚本基于 `scripts/install/install_common_parser.sh`，支持以下常用选项（来自 `help_info()`，可用 `--help` 查看，`install_common_parser.sh:1685-1798`）：

| 选项 | 说明 |
|------|------|
| `--install-path=<path>` | 指定安装路径 |
| `--install_for_all` | 安装为所有用户可用（开放 other 权限；通常 root 安装时使用） |
| `--chip=<chip>` | 按芯片过滤安装内容（匹配指定芯片或芯片为 all 的文件） |
| `--feature=<feature>` | 按特性过滤安装内容，默认 all（匹配指定特性或特性为 comm 的文件） |
| `--feature-exclude-all` | 切换特性模式：仅安装匹配指定特性的文件 |
| `--version=<ver>` / `--version-dir=<dir>` / `--version-file=<file>` | 指定版本信息（推荐用 `--version-file`） |
| `--username=<u>` / `--usergroup=<g>` | 指定安装属主/属组 |
| `--docker-root=<path>` | 指定 docker 根路径（安装路径不含该前缀时使用） |
| `--setenv` | 将 `source setenv.<shell>` 写入 rc 文件 |
| `--set-cann-uninstall` | 将卸载命令注册到 `cann_uninstall.sh`（多版本安装时常用） |
| `--custom-options=<args>` | 传给组件自定义安装脚本的参数 |
| `-h` / `--help` | 打印帮助 |

**交互式安装**（会提示输入安装路径，适合首次使用）：

```bash
chmod +x cann-runtime_9.1.0_linux-x86_64.run
./cann-runtime_9.1.0_linux-x86_64.run
```

**静默安装**（适合脚本化部署）：

```bash
# 静默安装到指定路径
./cann-runtime_9.1.0_linux-x86_64.run --install-path=/usr/local/Ascend

# root 安装并对所有用户可用（推荐生产环境）
sudo ./cann-runtime_9.1.0_linux-x86_64.run --install-path=/usr/local/Ascend --install_for_all

# 普通用户安装到自己的家目录
./cann-runtime_9.1.0_linux-x86_64.run --install-path=$HOME/Ascend

# 查看帮助
./cann-runtime_9.1.0_linux-x86_64.run --help
```

> **注意**：开放给所有用户的选项是 `--install_for_all`（下划线），**不是** `--install-for-all`。写错会报 `Unrecognized input options`（`install_common_parser.sh:2275-2278`）。

### 3.2 安装 rpm 包

rpm 包安装前缀由打包时的 `CPACK_PACKAGING_INSTALL_PREFIX` 决定（默认 `/usr/local/Ascend/cann-<version>`）。

```bash
# 安装
sudo rpm -ivh cann-runtime_9.1.0_linux-x86_64.rpm

# 升级安装（覆盖旧版本）
sudo rpm -Uvh cann-runtime_9.1.0_linux-x86_64.rpm
```

安装时 rpm 会自动执行内置的 `postinst` 脚本，内部走与 `.run` 相同的 filelist 安装逻辑。包间依赖由 `Requires` 字段声明，缺失依赖时 rpm 会提示。

### 3.3 安装 deb 包

```bash
# 用 dpkg 安装（不会自动解决依赖，需手动处理缺失依赖）
sudo dpkg -i cann-runtime_9.1.0_linux-x86_64.deb

# 用 apt 安装（自动解决声明在 Depends 中的依赖，推荐）
sudo apt install ./cann-runtime_9.1.0_linux-x86_64.deb
```

安装时 deb 会自动执行内置的 `postinst` 脚本，内部走与 `.run` 相同的 filelist 安装逻辑。

### 3.4 安装流程（执行了什么）

安装由 `install_common_parser.sh` 中的 `version_install()` 驱动（`install_common_parser.sh:1469-1530`），核心步骤：

1. **创建目录**（`do_create_dirs`）：解析 `filelist.csv` 中的 `mkdir` 动作，创建安装目录树并设置目录权限，处理包内软链接
2. **拷贝文件**（`do_copy_files`）：按 `copy`/`copy_entity`/`move` 动作拷贝文件，源端预先按权限规范设置权限位（`cp -af` 保留）；标记为 `configurable=TRUE` 的配置文件若已存在则**不覆盖**（保护用户已有配置）
3. **前置检查脚本**（`add_prereq_check`）：生成 `bin/prereq_check.<shell>`，供环境前置校验
4. **组件自定义安装**（`package_custom_install`）：执行 `<package>_custom_install.sh`（若存在）
5. **统一修正权限**（`set_install_permissions`）：按 [第五节](#五安装权限规范) 规范统一目录、Python 产物、安装脚本、`opp/built-in` 目录权限（`install_common_parser.sh:1127-1169`）
6. **注册卸载信息**（`add_cann_uninstall_package`）：将子包卸载命令写入安装根目录的 `cann_uninstall.sh`，并维护 `latest/version.cfg` 版本状态

### 3.6 安装后验证

安装完成后，建议做以下检查确认安装成功：

```bash
# 1. 查看安装目录是否存在
ls <安装根路径>/

# 2. 查看 version.cfg 版本状态（多版本场景）
cat <安装根路径>/latest/version.cfg

# 3. 查看库文件是否就位（以 runtime 为例）
ls <安装根路径>/<版本目录>/<arch>-linux/lib64/ 2>/dev/null

# 4. 查看卸载脚本是否生成
ls <安装根路径>/cann_uninstall.sh
ls <安装根路径>/<版本目录>/share/info/<package>/script/uninstall.sh

# 5. 验证权限是否符合预期（见第五节）
stat -c "%a %U:%G %n" <安装根路径>/cann_uninstall.sh   # 应为 750 或 744
```

---

## 四、卸载操作

### 4.1 卸载 .run 安装的内容

每个子包安装时会在 `share/info/<package>/script/` 下生成 `uninstall.sh`，并注册到安装根目录的总卸载脚本 `cann_uninstall.sh`（权限 750/744，仅安装者可执行）。

```bash
# 方式一：运行总卸载脚本（卸载所有已注册子包，推荐）
<安装根路径>/cann_uninstall.sh

# 方式二：运行单个子包卸载脚本（仅卸载该子包）
<安装根路径>/<版本目录>/share/info/<package>/script/uninstall.sh

# 卸载脚本同样支持 --install-path 等选项
./uninstall.sh --install-path=/usr/local/Ascend
```

> **注意**：`cann_uninstall.sh` 仅安装者本人或 root 可执行。若提示权限不足，请用安装时使用的账号或 root 运行。

### 4.2 卸载 rpm / deb 包

```bash
# rpm
sudo rpm -e cann-runtime

# deb（任选其一）
sudo dpkg -r cann-runtime
sudo apt remove cann-runtime
```

卸载时 rpm/deb 会自动执行内置的 `prerm` 脚本，内部走与 `.run` 相同的卸载逻辑。

### 4.3 卸载流程

卸载由 `version_uninstall()` 驱动（`install_common_parser.sh:1533-1588`）：

1. **恢复权限**（`do_create_dirs resetmod`）：先恢复目录写权限以便删除，清理 `db.info` 中该包的块信息
2. **清理前置检查脚本**（`del_prereq_check`）
3. **组件自定义卸载**（`package_custom_uninstall`）
4. **删除文件和目录**（`do_remove`）：按 `filelist.csv` 反向删除文件、软链接、目录
5. **更新版本状态**：从 `latest/version.cfg` 移除该包的版本记录

> **保护机制**：标记为 `configurable=TRUE` 且被用户修改过的配置文件会**跳过删除**（通过 sha256 校验判断是否被修改，`install_common_parser.sh:1215-1221`）。这是为了防止卸载时误删你已定制的配置。卸载后若发现配置文件残留，属预期行为，可手动删除。

---

## 五、安装权限规范

安装权限遵循最小权限原则（详见 `docs/install/permissions.md`），由 `set_install_permissions()` 统一设置（`install_common_parser.sh:1127-1169`）。

**三种安装场景**：

| 场景 | 安装者 | 可用范围 | 触发方式 |
|------|--------|----------|---------|
| 场景1 | root | 系统所有用户 | root 用户安装 |
| 场景2 | 普通用户 | 仅安装者及属组成员 | 普通用户默认安装 |
| 场景3 | 普通用户 | 系统所有用户 | 普通用户加 `--install_for_all` |

**权限对照表**（场景1 与场景3 权限值相同，仅属主/属组不同；场景1 为 `root:root`，场景3 为安装者:属组）：

| 资源类型 | 场景1 / 场景3 | 场景2 | 说明 |
|----------|--------|--------|------|
| 所有目录 | 755 | 750 | 目录需 `x` 位以保障遍历可达 |
| 可执行程序（ELF 二进制） | 755 | 750 | — |
| Python/Shell 脚本（.py/.sh） | 755 | 750 | 统一可执行 |
| 安装/卸载脚本 install.sh / uninstall.sh | 744 | 740 | 仅安装者可执行，防非授权卸载 |
| 其他文件（.so / .a / 头文件 / 配置 / 文档） | 644 | 640 | 无 `x` 位（库通过读取加载） |
| 符号链接 | 777 | 777 | 内核默认 |
| `opp/built-in` 目录 | 555 | 550 | 只读遍历 |

**设计依据要点**：
- **库文件不给 `x` 位**：`.so`/`.a` 通过 `ld.so`/`dlopen()` 读取加载，不调用 `execve()`，按最小权限不给 `x` 位
- **目录需要 `x` 位**：Linux 中目录 `x` 位是"遍历/进入"权限，路径上每级目录都需 `x` 位
- **`.py`/`.sh` 统一可执行，`.pyc` 不给 `x` 位**：`.pyc` 是解释器 import 加载的字节码，属数据文件
- **不采用防篡改（去除写位）设计**：对 root 无效，对普通用户仅防误操作，且卸载需"解锁/重锁"，代价大于收益

---

## 六、端到端示例

下面用一个完整示例串联打包到安装的全流程，帮助你快速上手。

### 6.1 场景：打包 runtime 组件并安装到生产服务器

**步骤 1：在构建机上打包**

```bash
# 进入 cmake 仓库目录（假设组件仓已在同级目录就位）
cd /path/to/CANN_TOP_DIR/cmake

# 打 runtime 的 .run 包（默认 Release、默认 run 类型）
sh build.sh --pkgs=runtime -v

# 产物位于
ls build_out/
# 预期看到：cann-runtime_9.1.0_linux-x86_64.run
```

**步骤 2：将产物拷贝到生产服务器**

```bash
scp build_out/cann-runtime_9.1.0_linux-x86_64.run user@prod-server:/tmp/
```

**步骤 3：在生产服务器上安装**

```bash
ssh user@prod-server
cd /tmp
chmod +x cann-runtime_9.1.0_linux-x86_64.run

# root 安装，对所有用户可用
sudo ./cann-runtime_9.1.0_linux-x86_64.run --install-path=/usr/local/Ascend --install_for_all
```

**步骤 4：验证安装**

```bash
ls /usr/local/Ascend/
cat /usr/local/Ascend/latest/version.cfg
ls /usr/local/Ascend/cann_uninstall.sh
```

**步骤 5：卸载（如需）**

```bash
sudo /usr/local/Ascend/cann_uninstall.sh
```

### 6.2 场景：打 deb 包并用 apt 安装

```bash
# 构建机：打 deb 包
sh build.sh --pkgs=runtime --pkg-type=deb
ls build_out/   # cann-runtime_9.1.0_linux-x86_64.deb

# 目标机：用 apt 安装（自动解决依赖）
sudo apt install ./cann-runtime_9.1.0_linux-x86_64.deb

# 卸载
sudo apt remove cann-runtime
```

---

## 七、常见问题

### 打包相关

**Q1：执行 `build.sh` 报 `error: --pkgs option is required`**
A：`--pkgs` 是必选参数，需指定要打包的组件名，如 `--pkgs=runtime`（`build.sh:216-219`）。

**Q2：报 `error: invalid --pkg-type 'xxx'`**
A：包类型必须是 `run`/`rpm`/`deb`/`deb,rpm`/`all` 之一（`build.sh:226-229`）。注意 `deb,rpm` 之间是逗号无空格。

**Q3：打包成功了但找不到产物**
A：产物在 `build_out/` 目录，不在 `build/`。若用 `--pkg-type=all` 会同时产出 `.run`/`.rpm`/`.deb` 三种。

**Q4：device 交叉编译失败**
A：device 编译需要 CANN 工具包提供 hcc 工具链。若仅需 host 产物，加 `--build_host_only` 跳过；若需要 device 产物，请确认 `ASCEND_CANN_PACKAGE_PATH` 指向有效的 CANN 工具包安装路径。完整排查见 `internals.md` 的"调试 superbuild"章节。

### 安装相关

**Q5：`.run` 安装后，其他用户无法访问**
A：普通用户默认为场景2（目录 750、文件 640），仅安装者及属组可访问。若需所有用户可用，安装时加 `--install_for_all`（root 或普通用户均可加此参数切换到场景3）。

**Q6：用 `--install-for-all`（连字符）报 `Unrecognized input options`**
A：选项名是 `--install_for_all`（**下划线**），不是连字符（`install_common_parser.sh:2223`、`2275-2278`）。

**Q7：卸载后发现配置文件仍残留**
A：标记为 `configurable=TRUE` 的配置文件，若被用户修改过（sha256 校验不一致），卸载时会跳过删除以保护你的定制配置。这是预期行为，确认无需保留后可手动删除。

**Q8：`cann_uninstall.sh` 提示权限不足**
A：该文件为 740/744，仅安装者本人或 root 可执行。请用安装时使用的账号或 root 运行。

**Q9：rpm/deb 安装路径不符预期**
A：安装前缀由打包时的 `CPACK_PACKAGING_INSTALL_PREFIX=/usr/local/Ascend/cann-<version>` 决定（`prepare.cmake:493`）。如需自定义，需在组件打包配置中修改后重新打包。

**Q10：多版本切换异常**
A：检查 `<安装根路径>/latest/version.cfg`，确认版本记录是否正确。必要时可手动清理失效版本记录，或通过 latest 管理器 `<安装根路径>/latest/var/manager.sh` 重建版本软链。

---

## 八、排错检查清单

遇到问题时，按以下清单逐项排查可解决大部分常见情况：

### 打包阶段

- [ ] `--pkgs` 是否已指定？包名是否在 [附录 B](#附录-b可打包的包名) 列表中（或对应组件仓已存在）？
- [ ] `--pkg-type` 取值是否合法（`run`/`rpm`/`deb`/`deb,rpm`/`all`）？
- [ ] CANN 组件仓是否已 clone 到 `CANN_TOP_DIR` 下，并与 cmake 仓库同级？
- [ ] 是否需要 device 产物？若不需要，是否加了 `--build_host_only`？
- [ ] 是否需要 device 产物且 CANN 工具包已安装？`ASCEND_CANN_PACKAGE_PATH` 是否正确？
- [ ] 加 `-v` 查看详细输出，定位具体失败步骤
- [ ] 查看 `build/CMakeCache.txt` 中 `CANN_DEPEND_PACKAGES` 是否符合预期

### 安装阶段

- [ ] `.run` 包是否已 `chmod +x`？
- [ ] 安装路径是否有写权限？（root 安装任意路径；普通用户需对目标路径有写权限）
- [ ] 是否需要多用户共享？若是，是否加了 `--install_for_all`（下划线）？
- [ ] 安装后查看 `<安装根路径>/latest/version.cfg` 确认版本记录
- [ ] 安装后查看 `<安装根路径>/cann_uninstall.sh` 是否生成
- [ ] 权限不符预期？用 `stat -c "%a %U:%G %n" <文件>` 核对，对照 [第五节](#五安装权限规范)

### 卸载阶段

- [ ] 用安装时的账号或 root 运行卸载脚本？
- [ ] 残留的配置文件是否是 `configurable=TRUE` 且被修改过？（属预期行为）
- [ ] 卸载后 `version.cfg` 中该包记录是否已移除？

---

## 九、术语表

| 术语 | 说明 |
|------|------|
| CANN | Compute Architecture for Neural Networks，面向昇腾 AI 处理器的计算架构生态 |
| cmake 仓库 | 本仓库，CANN 的公共构建/打包/安装框架，本身不产出制品 |
| CANN_TOP_DIR | 所有 CANN 组件仓的共同父目录，框架自动解析为 cmake 仓库的父目录 |
| superbuild | 多仓联合编译的元构建工程，入口为 `superbuild/CMakeLists.txt` |
| host 侧 | 在主机（x86_64）上运行的组件代码 |
| device 侧 | 在设备（aarch64）上运行的组件代码，通过交叉编译生成 |
| staging 目录 | 打包时的临时安装目录，`.run` 包由此目录打包生成 |
| `filelist.csv` | 文件清单，驱动安装/卸载的建目录、拷贝、设权限、删除等动作 |
| `version.cfg` | 版本状态配置文件，记录各包的 running/installed 版本 |
| latest 管理器 | 管理多版本软链的工具，位于 `<安装根路径>/latest/var/manager.sh` |
| `cann_uninstall.sh` | 总卸载脚本，位于安装根目录，聚合所有子包的卸载命令 |
| makeself | 将目录打包为自解压 `.run` 脚本的工具 |
| CPack | CMake 的打包工具，负责生成 `.rpm`/`.deb` 等格式 |
| 组件（component） | 一个包可包含多个组件，用于打包时区分 `COMPONENT` |

---

## 附录 A：filelist.csv 字段说明

`filelist.csv` 由打包阶段生成（`scripts/package/filelist.py` 的 `FileItem`），安装/卸载阶段由 `install_common_parser.sh` 解析。每行描述一个动作，主要字段：

| 字段 | 说明 |
|------|------|
| `operation` | 动作类型：`mkdir`（建目录）/ `copy`（拷贝）/ `copy_entity`（拷贝实体目录）/ `move`（移动）/ `del`（删除） |
| `relative_path_in_pkg` | 包内源路径 |
| `relative_install_path` | 安装目标相对路径 |
| `permission` | 权限位（如 750） |
| `owner_group` | 属主:属组 |
| `softlink` | 安装后需创建的软链接路径（NA 表示无） |
| `pkg_inner_softlink` | 包内软链接路径（NA 表示无） |
| `configurable` | 是否为配置文件（`TRUE` 时安装不覆盖、卸载不删除被修改的文件） |
| `hash_value` | 文件 sha256，用于卸载时判断配置文件是否被修改 |
| `block` | 块名，用于 `db.info` 块复用管理 |
| `chip` | 适配芯片集合（配合 `--chip` 过滤） |
| `feature` | 特性集合（配合 `--feature` 过滤） |

---

## 附录 B：可打包的包名

包名到源码目录的映射定义在 `superbuild/config.cmake`（完整列表与组件明细见 `getting-started.md` 的"可用的包名"）：

| 包名 | 源码目录 |
|------|---------|
| `npu-runtime` | `runtime` |
| `asc-devkit` | `asc/asc-devkit` |
| `asc-tools` | `asc/asc-tools` |
| `ge-executor` | `ge` |
| `ge-compiler` | `ge` |
| `dflow-executor` | `ge` |
| `hcomm` | `hcomm`（默认同名） |

> 未在映射中定义的包名，默认使用包名本身作为目录名。新增包的步骤见 `internals.md` 的"扩展指南"。

---

## 附录 C：参考文档与反馈

**参考文档**

- 联合编译使用指南：`docs/superbuild/getting-started.md`
- superbuild 构建机制：`docs/superbuild/internals.md`
- 安装权限规范：`docs/install/permissions.md`
- 整体架构：`docs/architecture.md`
- 仓库 README：`README.md`

**反馈与贡献**

- 本说明书基于 `cann/cmake` 仓库 master 分支源码分析整理，关键结论已与代码逐行核对。如仓库代码更新导致文档与实际不符，欢迎反馈。
- 发现文档错误或有改进建议，请通过仓库 issue 反馈。
- 贡献指南见 `CONTRIBUTING.md`。

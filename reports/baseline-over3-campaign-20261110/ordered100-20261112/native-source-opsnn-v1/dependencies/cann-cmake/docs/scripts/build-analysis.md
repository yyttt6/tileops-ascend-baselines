<!-- -----------------------------------------------------------------------------------------------------------
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
----------------------------------------------------------------------------------------------------------- -->

# IWYU Log Parser

> 脚本位置：`scripts/build_analysis/iwyu_log_parser.py`

解析 IWYU (Include What You Use) 日志，自动识别和修复 C/C++ 头文件依赖问题。

## 功能特性

- **解析 IWYU 日志**：识别需要添加、移除的头文件依赖
- **生成分析报告**：汇总每个文件的依赖问题，支持导出到文件
- **自动修复源码**：注释掉冗余的用户头文件
- **安全清理**：永久删除已注释的头文件行（仅删除脚本生成的标记注释，不会误删用户原有注释）
- **智能过滤**：仅修改源文件（.c/.cpp/.cc），跳过头文件；仅处理用户头文件（`"header.h"`），保留系统头文件（`<header.h>`）
- **自动去重**：输出报告中自动合并重复项
- **参数互斥检查**：防止 `--fix` 和 `--clean-commented` 同时使用，避免误操作
- **预览模式**：`--dry-run` 参数可预览将要变更的文件，不实际修改源码
- **异常处理**：文件读写失败时输出警告而非崩溃，继续处理其他文件
- **原子写入**：文件修改采用原子写入策略（先写临时文件再替换），避免写入中途崩溃导致文件损坏
- **重复运行保护**：`--fix` 不会对已标记的注释行重复注释

## 前置要求

- Python 3.7+
- 已安装并配置好 IWYU 工具

安装 IWYU：
```bash
# Ubuntu/Debian
sudo apt-get install iwyu

# macOS
brew install iwyu

# 从源码编译
# 参考: https://github.com/include-what-you-use/include-what-you-use
```

## 使用方法

### 1. 生成 IWYU 日志

首先对代码库运行 IWYU 工具并保存日志：

```bash
# 对单个文件
iwyu_tool.py -p build/ source.cpp > iwyu.log

# 对整个项目
iwyu_tool.py -p build/ > iwyu.log

# 使用编译数据库
include-what-you-use -Xiwyu --mapping_file=... source.cpp > iwyu.log 2>&1
```

### 2. 生成分析报告

解析日志并查看依赖问题：

```bash
# 从文件读取
python iwyu_log_parser.py iwyu.log

# 从管道读取
iwyu_tool.py -p build/ | python iwyu_log_parser.py -

# 导出报告到文件
python iwyu_log_parser.py iwyu.log --report-out report.txt
```

### 3. 自动修复（推荐流程）

采用两阶段修复，确保安全可控：

**阶段一：注释冗余头文件**
```bash
python iwyu_log_parser.py iwyu.log --fix
```

此操作会将 `.c/.cpp/.cc` 文件中的冗余用户头文件注释掉（添加 `[iwyu-fix]` 标记以区别于用户原有注释）：
```cpp
// 修改前
#include "redundant.h"
#include "needed.h"

// 修改后
// [iwyu-fix] #include "redundant.h"
#include "needed.h"
```

**阶段二：验证并清理**

1. 编译项目验证修改正确性
2. 确认无误后，永久删除脚本标记的注释行：
```bash
python iwyu_log_parser.py iwyu.log --clean-commented
```

> `--clean-commented` 仅删除带有 `[iwyu-fix]` 标记的注释行，用户原有的 `// #include ...` 注释不会被误删。

### 预览模式

在执行 `--fix` 或 `--clean-commented` 前，可使用 `--dry-run` 预览将要变更的文件，不实际修改源码：

```bash
# 预览哪些文件将被注释冗余头文件
python iwyu_log_parser.py iwyu.log --fix --dry-run

# 预览哪些文件将被清理注释行
python iwyu_log_parser.py iwyu.log --clean-commented --dry-run
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `log_input` | IWYU 日志文件路径，输入 `-` 从标准输入读取 |
| `--report-out FILE` | 将分析报告导出到指定文件 |
| `--fix` | 注释掉冗余的用户头文件（仅修改 .c/.cpp/.cc 文件） |
| `--clean-commented` | 永久删除脚本生成的标记注释行（仅删除含 `[iwyu-fix]` 标记的行） |
| `--dry-run` | 预览模式：与 `--fix` 或 `--clean-commented` 配合使用，不实际修改源码（同时跳过 `--report-out` 文件写入）。注意：dry-run 仍会向标准输出打印完整分析报告，随后再打印变更文件清单 |

**注意**：`--fix` 和 `--clean-commented` 不能同时使用，必须分两步执行。

## 工作原理

### 日志解析

脚本解析 IWYU 输出的三种区块：

1. **should add 区块**：需要添加的头文件
   ```
   /path/to/file.c should add these lines:
   #include "missing.h"
   ```

2. **should remove 区块**：需要移除的冗余头文件
   ```
   /path/to/file.c should remove these lines:
   - #include "redundant.h"  // lines 10-10
   ```

3. **full include-list 区块**：文件应包含的完整头文件列表
   ```
   The full include-list for /path/to/file.c:
   #include "needed.h"
   ```

### 修复策略

1. **仅修改源文件**：只处理 `.c`、`.cpp`、`.cc` 文件，从不修改 `.h`、`.hpp`、`.hh` 头文件
2. **仅注释用户头文件**：只注释双引号形式的用户头文件 `"header.h"`，保留尖括号形式的系统头文件 `<header.h>`
3. **两阶段修复**：先注释，验证后删除，避免不可逆操作
4. **标记防误删**：`--fix` 注释时添加 `[iwyu-fix]` 标记，`--clean-commented` 仅清理带标记的行，不会误删用户原有注释
5. **自动去重**：同一文件的多次 IWYU 建议会自动合并

### 报告输出格式

```
==========================================================================================
IWYU 头文件依赖分析报告
==========================================================================================

【文件】/path/to/file.c
  [1] should add 建议新增头文件(2):
      + #include "missing1.h"
      + #include "missing2.h"
  [2] should remove 冗余头文件(3):
      - #include "redundant.h"
      - #include "unused.h"
  [3] The full include-list 全部头文件(5):
      #include "needed.h"
      #include "required.h"
----------------------------------------------------------------------
==== 全局汇总统计 ====
解析文件总数(已去重): 42
可修复源码(.c/.cpp/.cc)数量: 38
建议新增头文件总行数(已去重): 15 (其中源文件: 12)
冗余待清理头文件总行数(已去重): 127 (其中源文件: 110)
==========================================================================================
```

## 注意事项

### 安全性

- **总是先生成报告**：执行 `--fix` 前先查看分析报告，确认 IWYU 建议合理
- **分阶段修复**：使用 `--fix` 注释而非直接删除，可随时撤销修改
- **预览模式**：使用 `--dry-run` 预览变更，确认无误后再实际执行
- **标记防误删**：脚本注释行带有 `[iwyu-fix]` 标记，`--clean-commented` 仅清理带标记的行，用户原有的 `// #include ...` 注释不受影响
- **重复运行安全**：对已标记 `[iwyu-fix]` 的注释行再次运行 `--fix` 不会产生双重注释
- **原子写入**：文件修改通过临时文件+替换完成，避免写入中断导致文件损坏
- **参数互斥保护**：`--fix` 和 `--clean-commented` 同时使用会报错并退出，防止误操作
- **异常容错**：文件读写失败（权限、编码等问题）时输出警告，继续处理其他文件
- **编译验证**：每次 `--fix` 后务必编译项目确认无错误
- **版本控制**：修复前确保代码已提交到 Git，方便回滚

### 限制

- 只处理源文件，不修改头文件（修改头文件影响范围大，需人工判断）
- 只注释用户头文件，系统头文件保持不变（系统头文件依赖关系复杂）
- 无法处理条件编译的头文件依赖（需人工检查）
- 无法处理宏定义的头文件包含（需人工检查）
- **不感知 C/C++ 词法上下文**：`--fix` 按行级 token 匹配，不跟踪块注释 `/* */` 和条件编译死代码块（如 `#if 0 ... #endif`）状态。若冗余头文件的 `#include` 行同时出现在这些块内，块内的行也会被一并注释。通常无害（注释死代码内的 include 不影响编译），但会破坏块注释结构、需人工复核

### IWYU 配置建议

使用映射文件提高 IWYU 分析准确性：

```bash
# 生成映射文件示例
cat > iwyu.imp <<EOF
[ { include: ["@public_header.h", "private", "<public_header.h>", "public"] } ]
EOF

# 使用映射文件
iwyu_tool.py -p build/ -- -Xiwyu --mapping_file=iwyu.imp
```

### 常见问题

**Q: `--clean-commented` 会误删我手动注释的 `// #include ...` 行吗？**

A: 不会。`--fix` 注释行时添加了 `[iwyu-fix]` 标记，`--clean-commented` 仅删除包含该标记的行。用户原有的 `// #include "xxx.h"` 注释不会被删除。

**Q: 为什么有些头文件没有被注释？**

A: 可能原因：
- 文件是 `.h`/`.hpp`/`.hh` 头文件，脚本会跳过
- 头文件是系统头文件 `<...>` 形式，脚本保留不修改
- IWYU 日志中没有标记该文件为冗余

**Q: 注释后编译失败怎么办？**

A: 
1. 从版本控制恢复文件
2. 检查 IWYU 建议是否准确，某些建议可能不适用于您的代码环境
3. 手动选择性修复，跳过有问题的建议

**Q: 为什么同时使用 --fix 和 --clean-commented 会报错？**

A: 两个参数必须分步执行：
```bash
# 错误用法
python iwyu_log_parser.py iwyu.log --fix --clean-commented

# 正确用法
python iwyu_log_parser.py iwyu.log --fix     # 第一步
python iwyu_log_parser.py iwyu.log --clean-commented  # 第二步（验证后）
```

这样可以确保先验证 `--fix` 的修改正确性，再执行不可逆的清理操作。

**Q: 重复运行 `--fix` 会产生双重注释吗？**

A: 不会。脚本会检测已标记 `[iwyu-fix]` 的注释行并跳过，不会重复注释。

**Q: 可以对单个文件处理吗？**

A: 可以，但**不能直接使用 `grep`**。IWYU 日志按块组织，每块由"块头行 + 若干 `#include` 内容行"组成，块间以空行分隔：

```
/path/to/target_file.cpp should add these lines:
#include "missing.h"
```

`grep "target_file.cpp"` 只会命中块头行（含文件名），而 `#include` 内容行不含文件名会被丢弃。解析器的状态机在块头行之后需要连续识别 `#include` 行来填充数据；仅有块头而无内容行时该文件数据为空，`generate_report` 会静默跳过，导致输出为空或误判无问题。

正确做法是使用 `awk` 段落模式（`RS=''`）按空行分段，提取包含目标文件的**完整块**：
```bash
awk -v RS='' -v ORS='\n\n' '/target_file\.cpp/' iwyu.log | python iwyu_log_parser.py -
```

**Q: 为什么有些文件处理失败？**

A: 可能原因：
- 文件权限不足（无读写权限）
- 文件编码异常
- 文件不存在或路径错误
- 文件被其他进程占用

脚本会输出警告信息，列出所有失败文件及原因，可手动处理这些文件或检查权限后重新运行。

## 示例工作流

完整的项目头文件优化流程：

```bash
# 1. 生成编译数据库
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build .

# 2. 运行 IWYU 并保存日志
iwyu_tool.py -p build/ > iwyu.log 2>&1

# 3. 查看分析报告
python iwyu_log_parser.py iwyu.log --report-out iwyu_report.txt

# 4. 预览将要注释的冗余头文件
python iwyu_log_parser.py iwyu.log --fix --dry-run

# 5. 注释冗余头文件
python iwyu_log_parser.py iwyu.log --fix

# 6. 编译验证
cmake --build build/

# 7. 预览将要清理的注释行
python iwyu_log_parser.py iwyu.log --clean-commented --dry-run

# 8. 确认无误后清理注释行
python iwyu_log_parser.py iwyu.log --clean-commented

# 9. 再次编译确认
cmake --build build/

# 10. 提交修改
git add -A
git commit -m "fix: remove redundant header includes based on IWYU analysis"
```

## 输出文件说明

脚本执行 `--fix` 和 `--clean-commented` 后会输出变更清单：

```
==== --fix 注释冗余自定义头文件 变更清单 ====
MODIFIED: /path/to/file1.c
MODIFIED: /path/to/file2.cpp
本次修改文件总数: 2

==== --clean-commented 永久清理注释行 变更清单 ====
CLEANED: /path/to/file1.c
CLEANED: /path/to/file2.cpp
本次清理文件总数: 2
```

如果部分文件因权限或编码问题处理失败，会输出警告：

```
[WARNING] 以下文件处理失败：
  - /path/to/file3.cpp: Permission denied
  - /path/to/file4.c: No such file or directory
```

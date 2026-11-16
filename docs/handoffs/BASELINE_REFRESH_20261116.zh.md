# Baseline 更新交接：代码与结果定位

仓库：`https://github.com/yyttt6/tileops-ascend-baselines`  
分支：`baseline/over3-refresh-20261116`  
代码、结果快照提交：`01e6bedec9e5cfd19d5b9f38b4cb0585948216d4`。本文随该分支的后续提交发布。

这份交接介绍本次 baseline 工作和材料位置。没有修改 CI/nightly，也没有执行网页发布。

## 采用清单补充（2026-11-16）

本交接的2119项仅是冻结覆盖结果，不能作为完整优化采用清单。已有历史优化与本轮原路径复测的衔接问题、逐项采用工作清单及核验命令，请先读 [采用清单补充交接](BASELINE_RECONCILIATION_20261116.zh.md)。旧记录保持不变；尚未完成采用核对的优化不能因覆盖完成而结项。

## 本次完成了什么

本轮冻结范围是 **2119 个 library/op/case 组合**，已全部取得合格 baseline 覆盖。历史另有89项，属于不同账本；首100项已包含在2119里，不能重复相加。此前剩余的69项，现在全部取得合格覆盖。

主要工作包括：恢复厂商原有运行栈和完整 API，修复注册、输入准备和包装入口问题；逐项审核原检查、全部输出、输入保护、实际 device 采样、源码/产物归属与整段 guard；对符合条件的相同输入合同明确复用已有完整 API 证据。替代库的 API 和原库实现分别记录，不把替代实现验收写成原库原生实现验收。

最后两组修复：

- **42项 golden 修正**：用户逐项授权，精确绑定 ordinal、key、参数、输入 payload/descriptor SHA。CPU FP64 两条独立计算路径复核，输出转回原 dtype；26项完成新的原生完整 API 实测，16项完成精确完整 API 证据复用。原 kernel、输入、被测精度及输出 checker/容差保持不变。修正规则不是对任意输入的全局 golden 替换。
- **6项复数输入保护兼容**：All/Any/CountNonzero 各 complex64/complex128。原 NPU IsClose 不支持复数；用户新授权后，仅在这六项独立进程内，将输入是否被修改的比较通过保位传输移到 CPU，继续调用同一原零容差/equal-NaN 比较函数。计时外做真实 NPU 转 CPU 的同值及实部/虚部/shape/dtype破坏检查，每项结束后恢复原检查器。原 NPU baseline、golden、输出检查和计时不变；六项共90个正式窗口、114次 baseline 调用均通过。

**合格覆盖不是最终倍率验收。** 当前有证据的同环境最终倍率结论仍为14项：13项低于3倍，1项保留原实现且仍≥3倍。其他覆盖记录不能因此被标成最终<3、原新对照完成或已经发布。

## 先读机器索引

在本仓库根目录执行，均为只读 CPU 命令，不导入 Torch/NPU：

```sh
git fetch origin baseline/over3-refresh-20261116
# 在独立 checkout/worktree 中读取该分支。
python3 scripts/over3_refresh.py summary
python3 scripts/over3_refresh.py case --ordinal 1292
python3 scripts/over3_refresh.py list --library ops-math --op GemmFwdOp
python3 scripts/over3_refresh.py verify
python3 scripts/verify_sources.py
```

| 位置 | 内容 |
| --- | --- |
| `catalog/baseline-over3-refresh-20261116/summary.json` | 分母、覆盖、42/6项完成情况、最终倍率结论数量、材料保留位置 |
| `catalog/baseline-over3-refresh-20261116/cases.json` | 2119项去重索引，共2136条合格实现/复用记录；每条包含目标 key、device_us、实际 API、来源 ordinal、协议/环境及证据 JSON 路径和 pointer |
| `catalog/baseline-over3-refresh-20261116/case-contracts.json` | 2066个 ordinal 的原准备元数据版本，保留参数、dtype/shape、输入/描述符哈希和原完整元数据 SHA |
| `catalog/baseline-over3-refresh-20261116/checkpoint-headers.json` | 1150份原 checkpoint 的紧凑头部、运行环境、guard、采样摘要和原文件 SHA/路径 |
| `catalog/baseline-over3-refresh-20261116/code-by-sha256.json` | 内容 SHA → 仓库内代码路径，方便按结果中的源码哈希找文件 |
| `catalog/results.json` | 新增独立 campaign `baseline-over3-refresh-20261116`，保留旧 campaign 原记录 |

`cases.json` 的 `qualified_routes` 保留全部合格记录，**没有跨运行环境取最小值作为统一推荐结果**。优先用 `[library, op, case]` 和完整输入合同定位；ordinal 只在本轮冻结清单内有意义。使用 `device_us`，`host_us_diagnostic` 仅供诊断。

部分摘要没有直接携带 runtime/protocol，索引保留空值，没有猜补；原文件路径、SHA和证据 pointer 可继续追溯。53项未收集到可导出的准备元数据，不代表重新生成过输入或补写过合同，按原证据和冻结 case 源码继续定位。

## 对应代码在哪里

以下以 `D=reports/baseline-over3-campaign-20261110/ordered100-20261112` 表示仓库内目录。

- `D/canonical-source/TileOPs/`：从一号实际任务工作区保存的冻结 benchmark、workload、manifest 和原 checker 源码。与仓库根目录较早的 `TileOPs/` 快照分开放置，不要混作同一源码版本。
- `D/*.py`：本次各 API 适配、检查、留存、审计、测量 worker 和历史控制器源代码。例如 `ops_nn_gemm10_api.py`、`ops_nn_gemm_pad4_api.py`、`math_bmm_transformer_alternative_api.py`、`math_grouped_nn_alternative_api.py`。
- `reports/baseline-over3-campaign-20261110/delivery/`：各组已整理的适配器和 `ROUTES.json`，与逐项索引中的精确验收范围一起读。
- `D/server2-alignment-20261113/`：环境入口脚本、来源/运行栈诊断，以及本次42项和6项的授权、预检、修复与最终结果。
- `D/restricted_profiler/`、`D/measured_correctness.py`、`D/compact_timing.py`：本轮测量、最后一次计时输出验证、原始记录摘要支持代码。

这些源文件保留原目录和源码字节，包括原换行/空白；许多路径仍指向原机器的 `/opt/...`。它们是源代码快照，**不是直接运行的安装入口**。不要 import `queue_*.py`/`worker_*.py` 来查询结果，它们可能在模块顶层派发工作。源代码版本也不能替代私有 native 包的构建和产物资格。

42项重点入口：

- `D/golden42_reference.py`、`D/golden42_bindings.py`
- `D/certify_golden42_cpu.py`、`D/GOLDEN42_CERTIFICATES.json`
- `D/GOLDEN42_ACTIVATION_AUDIT.json`
- `D/server2-alignment-20261113/GOLDEN42_ALL42_QUALIFIED_RESULT_20261116.json`

六项重点入口：

- `D/complex_input_preservation_compat_proposal.py`：获授权的原比较转 CPU helper，内容哈希保留。
- `D/complex6_input_preservation.py`：限定六项、进程内安装/恢复及真实转移破坏检查。
- `D/worker_complex6_original.py`、`D/complex6_cached_fidelity.py`
- `D/summarize_complex6_original.py`
- `D/server2-alignment-20261113/COMPLEX6_USER_AUTHORIZATION_20261116.json`
- `D/server2-alignment-20261113/COMPLEX6_ALL_RESOLVED_RESULT_20261116.json`

## 如何追溯一条结果

1. 在 `cases.json` 找目标 key，读取对应 `qualified_routes`，留意实际 API 和 `source_ordinal/source_key`。例如目标309明确复用了777的 Transformer完整接口，不能写成 Math 原生实现重新测量。
2. 打开 `evidence_path`，按 `json_pointer` 读取原合格记录。前100项以 `D/RESULT.json` 的 `selections` 为入口，其余由 `D/ROLLOUT_LEDGER.json` 与具名 `RESOLVED_*.json` 覆盖层去重。不要将旧 ledger 的失败状态当作最终状态；修复和合格覆盖层单独保留。
3. 用 checkpoint 名称、SHA及 `checkpoint-headers.json` 查原 guard、环境、采样记录；用源码 SHA 和 `code-by-sha256.json` 找适配文件。哈希未匹配的历史版本不能默认为当前文件。
4. 如果需要最终同环境倍率结论，另读 `D/FINAL_SAME_ENV_CLOSURES_CURRENT.json`；不能直接用冻结 `TASKS.json` 中的历史倍率充当更新倍率。

## 完整证据保留位置

Git 中上传的是文本代码、合格结果摘要、索引和哈希。没有上传 `.pt` 输入/golden 张量、私有 `.so/.o`、完整 profiler raw、cache 或大归档。

- 本地完整任务：`/root/dyq/workspace/reports/baseline-over3-campaign-20261110/ordered100-20261112/`
- 一号：`openlibing-DevEnv-661495`
- 二号：`openlibing-DevEnv-055409`
- 两机任务根：`/opt/baseline-over3-20261110/ordered100-20261112/`
- 本地归档镜像：上述本地任务目录的 `remote/runs/<tag>/`；原 checkpoint、`<tag>.tar.gz` 和各项 guard 回执保留。
- private native 运行栈：`/opt/baseline-over3-20261110/server2-aligned-stack/`，具体路径/源码与产物哈希读逐项证据。

紧凑 checkpoint 头和准备元数据的 SHA 字段指向**原完整文件**，不能拿压缩后的 JSON 重新算 SHA 并要求相等。记录中的原绝对路径不是 GitHub 下载链接。

## 本次导出校验

2119个 key/ordinal 唯一且都有合格 device 值；本轮8094份快照文件内容哈希通过，仓库34896条源文件清单均通过并已入 Git 索引。发现原仓库清单已有3份 `baseline-shared28/.../perf/profiles/` 文本文件被忽略而未入库，已按原清单 SHA 逐字补齐。原工作区尚未提交的 `docs/BASELINE_ADJUSTMENT_METHODS.zh.md` 没有包含在本次提交中。

# Baseline 性能调优交接（在自己的机器上开发）

分支：`baseline/perf-review-20261117`。从已发布的
`baseline/over3-refresh-20261116` / `7a12f33` 建立。

**从 `review/` 开始修改和运行。** 仓库根目录 `TileOPs/`、旧 `reports/`
和旧 `tools/worker*` 是多个时期的源码档案，不能当成统一的当前安装或启动入口。
本分支不需要原机器、SSH、原进程或内部巡检服务；本文不包含内部用卡协议。

## 1. 现在交给你的是什么

目标是让同一 case 的完整 baseline 更快，使
`R = baseline_device_mean_us / tileops_device_mean_us <= 3.0`。
R 越大，baseline 相对于 TileOPs 越慢；R=2 表示 TileOPs 是 baseline 的2倍速度。

- 冻结范围为 **665 个去重 `(op, case)`**。原网页4314个>3x的库/case pair里，
  有666个并列/独占最快 pair，对应这665个case。其他库较慢的pair不是这批分母。
- 暂停时，**178/665 在原机器、原合同下完成≤3x验收，487项未完成目标**。
  这是历史资格，不是你机器上的验证，也不是网页/CI已采用这些新结果。
- 旧2119是上一轮合格覆盖数量，不是2119项均优化到3x。不要用它作为本轮进度。
- 我方设备调优保持暂停。你可以在本分支独立修改、测量、提交，不依赖我方继续跑。

| 路径 | 用途 |
| --- | --- |
| [`review/catalog/WORKLIST.csv`](../../review/catalog/WORKLIST.csv) | 全部665项，当前结果/待处理状态、原最快库、候选入口 |
| [`review/catalog/ACCEPTED_SELECTIONS.json`](../../review/catalog/ACCEPTED_SELECTIONS.json) | 178项逐case选中路线、参数、历史均值/倍率、独立进程次数 |
| [`review/baselines/`](../../review/baselines/) | 选中实现的27份源码及额外未达标路线；**主要编辑位置**，保留原代码字节/SHA |
| [`review/factories.py`](../../review/factories.py) | 可编辑的选中实现、原 eager、原 vendor 与本机路径绑定入口 |
| [`review/bench.py`](../../review/bench.py) | 本机生成输入、完整正确性、原 isolated15 device 测量入口 |
| [`review/compare.py`](../../review/compare.py) | 离线重放全部raw、同合同比较、3x边界与独立复测判断 |
| [`review/diagnose.py`](../../review/diagnose.py) | 15窗完整任务拆分，定位哪段占时间 |
| [`review/frozen/torch20261105/`](../../review/frozen/torch20261105/) | 实际采用过的原 benchmark/workload/manifest 源码，保持只读 |
| [`review/evidence/`](../../review/evidence/) | 历史最终闭环及采样/源码身份摘要；完整raw不在Git |
| [`review/catalog/EXPORT.json`](../../review/catalog/EXPORT.json) | 导出版本、源码SHA、历史资格和便携性边界 |

27份实现涵盖 INT4量化、INT8量化/反量化、W4A8/INT8/FP8 GEMM、
RMS+量化、attention merge、TopK/TopP及GDN。未达标case没有被删除或标为不可优化。
现有原完整 vendor provider 在旧 canonical snapshot 的
`benchmarks/ascend_all_pairs/<library>/`，可以继续审其完整API/包装。

## 2. 环境和代码版本

```bash
git clone --branch baseline/perf-review-20261117 \
  https://github.com/yyttt6/tileops-ascend-baselines.git
cd tileops-ascend-baselines
python3 -m review.doctor
python3 -m review.bench catalog --op TopKMaskFwdOp

# 单独保留不修改的TileOPs性能对手；不要拿根目录旧快照代替。
git clone https://github.com/yyttt6/TileOPs.git ../TileOPs-public
git -C ../TileOPs-public checkout --detach 62c5fa4d1f80b5325c6632aa23622ec2d6c3f2c9
```

先在本机准备能运行Ascend的Python/CANN/编译环境。固定对手的工具链要求见
`../TileOPs-public/config/ascend950-toolchain.json`、
`../TileOPs-public/docs/ascend950/toolchain.md` 和实际工具链检查脚本。
本分支没有自动安装驱动、修改全局环境或下载私有二进制的脚本。

历史CI/public profile为Torch2.13.0+cpu / torch-npu2.13.0rc1 / SDK9.2；
vLLM/SGL的原厂隔离profile为Torch2.10.0+cpu / torch-npu2.10.0.post4。
参考compiler9.1可属于经过核验的隔离profile，不是仅看版本号就判错。
**不要混用一个Torch的Python包和另一个Torch的native扩展；也不要强行把所有库装成一套版本。**
按对应源码pin、build入口和ABI准备本机依赖，记录实际版本、flags与产物SHA。
公开对手的工具链核验是另一条独立要求。

只需要Torch现成API的例子可先不初始化第三方submodule。
需要某库时按 `dependencies.lock.json` 选择性执行：

```bash
git submodule update --init ascend_baselines/flash-linear-attention-npu
```

`patches/`记录历史改动，不要无差别自动应用后仍声称用了原厂kernel。
原库某API没编译进本机包时，须编译原完整API或明确报告部署缺口；
包里有Python函数名不证明对应native算子已构建、注册或能运行。

## 3. 第一个完整例子：TopK

下面命令在本仓库根目录执行。使用你自己的环境启动Python，不需要原机器路径。
例子选 `TopKMaskFwdOp / deepseek-v3-b17-mixed-bfloat16`：
历史选中实现是 `review/baselines/top_k_storage_sort_v1.py:top_k_mask`。

### 生成一次真实输入

```bash
python -m review.bench prepare \
  --op TopKMaskFwdOp --case deepseek-v3-b17-mixed-bfloat16 \
  --output local-runs/topk-input
```

这一步使用冻结源码中的原输入生成器，在NPU生成一次，然后保存完整
`inputs.pt` 和 `contract.json`。seed按原 `SHA256(op::case)` 规则。
这是**本机新生成的共同输入**；由于历史完整输入张量未上传，不能说它已证明
与历史`.pt`逐字节一致。所有后续角色必须读取这一个文件，不能各自重新抽随机输入。
不要手改里面的数值、dtype、shape或golden来争取倍率。

### 量 baseline 和固定 TileOPs

```bash
# 默认选择已验收实现或已记录的当前候选；本例是accepted selection。
python -m review.bench measure \
  --source-root review/frozen/torch20261105 \
  --inputs local-runs/topk-input --side baseline \
  --output local-runs/topk-baseline-1

# 同一inputs.pt，固定的public Op + public完整binder；不同环境可换Python启动。
python -m review.bench measure \
  --source-root ../TileOPs-public \
  --inputs local-runs/topk-input --side tileops \
  --output local-runs/topk-tileops-1

python -m review.compare \
  --baseline local-runs/topk-baseline-1/result.json \
  --tileops local-runs/topk-tileops-1/result.json \
  --output local-runs/topk-comparison-1.json
```

每个输出目录必须新建，不覆盖旧结果。脚本异常时保留`FAILURE.txt`和已产生的raw。
`review.compare`会离线重放两边全部15窗raw，并输出实际均值/倍率。
**`numeric_target_met=true`与`accepted=true`分开：** 还需要核实native/JIT执行身份、
物理设备与运行profile、完整timed body及测量无干扰，不能只凭脚本退出0验收。

```bash
python -m review.identity_template \
  --baseline local-runs/topk-baseline-1/result.json \
  --tileops local-runs/topk-tileops-1/result.json \
  --output local-runs/identity-review.json
# 检查实际证据后填写reviewer、物理设备身份、notes_and_evidence_paths和对应检查项。
# 模板默认false。不要为了得到accepted而跳过证据检查。
python -m review.compare \
  --baseline local-runs/topk-baseline-1/result.json \
  --tileops local-runs/topk-tileops-1/result.json \
  --identity-review local-runs/identity-review.json \
  --output local-runs/topk-reviewed.json
```

若比值在 **[2.9,3.0]**，必须用**新Python进程、同输入**独立复测baseline；
传入两份result，按较慢进程的15次均值除以同一合格分母。复制JSON不算独立复测。
分母受代码、输入、栈或机器变化影响时也要重建；不能一直复用不再可比的旧分母。

## 4. 怎么修改 baseline

直接修改选中的 `review/baselines/*.py`。`catalog/ACCEPTED_SELECTIONS.json`
绑定旧源码资格，**修改代码后不得继续把旧结果称为新版本验证**。

```bash
python3 -m review.doctor --allow-edited-baselines
```

新case/新路线可写一个本机factory，例如：

```python
# my_baseline.py
def build(case):
    # case.params包含完整原参数；这里只构造输入无关的API常量。
    def run(*actual_inputs):
        # 每次从actual_inputs执行完整API，返回全部必需输出。
        ...
    return run
```

用 `--factory my_baseline.py:build` 传给`review.bench measure --side baseline`。
也可以返回原 `benchmarks.api.Implementation(run, reset=...)`。
通用runner拒绝`noncomparable_reason`及替换参数的`impl.args`；私有参数/别名/状态协议
需要显式审过的binder，不能靠通用脚本静默替换。
对照原eager可用 `--factory review.factories:original`；这是真实计时的原参考运算，
不是读取golden或旧耗时。原vendor可用 `REVIEW_LIBRARY=<library>` 和
`--factory review.factories:vendor`，但本机要先构建/安装对应pin和完整API。

TopP原FLA cumsum可以设置`REVIEW_FLA_ROOT`，或把
`REVIEW_FLA_CUMSUM_SOURCE`指到原pin的`fla/ops/triton/triton_core/cumsum.py`；
会核对原SHA。vLLM输出view那一项的原specialization仍依赖历史路径/来源表，
已作为原源码保留，**不伪称一键便携**：需要在本机明确重绑provider与来源表，
使用custom factory，重新检查完整输入/输出与状态布局。

### 允许改什么

- 寻找更贴近目标语义的现成完整API，修复明显错误/多余的包装和注册入口。
- 删除有证明的identity copy、零padding、重复物化；保留必需的输入保护和输出语义。
- Torch eager可使用完整数学表达式/现成API组合；第三方原生路线保持原完整API及
  逐项允许的包装/原生调用数边界，不能把“≤3个原生算子”机械套成“只能3个Torch运算”。
- 编译缓存、常量表、输入无关的API参数可复用；每次实际输入相关的pack、scale、gate、
  copy、layout转换、fallback predicate与全部输出都在完整调用里真实执行并计时。

保持kernel及compiler代码/调度/编译flags、计算精度、输入和checker/容差不变。
不能缓存model输入/权重派生结果或输出，不能删辅助输出、只量microkernel、换小输入，
也不能用FP16/BF16窄化FP32状态来过线。历史经过逐值证明的无损存储技巧仅在其限定域有效。
torch.compile等另立基线类别时需单独授权与合同，不能悄悄替换原库类别。

## 5. 正式测量方式

本分支将新比较固定到 **`ascend950-isolated-ffts-v3`**，两边同口径。
冻结timer和full32 parser在`review/support/tools/`，源码SHA在EXPORT中。

1. 记录1次冷调用供编译/冷启动诊断；3次warmup；**15次正式采样取算术均值**。
2. 冷调用和warmup同样按原timer执行reset、原 **1GiB flush**与同步。
   15次正式采样各自独立打开profiler窗口，执行完整callable、同步，再结束窗口；
   reset/flush在窗口外，不混进被测API。
3. device耗时为完整解析后不同设备任务跨度之和；AIC/AIV按任务身份去重，
   **真实AICPU、必要copy/pack/转换也要计入一次**。缺任务、丢trace、无法归属即不验收。
4. 保存全部15样本、全部原raw及SHA、parser源码身份；不删慢样本、不取最小值、
   不拿host整段除以15，不用Event、host wall或profiler总时长当正式device耗时。
5. 冷调用/CPU启动/JIT编译/golden生成/归档耗时单列。1GiB flush是约定操作，
   不意味着已经证明所有目标数据处于完全冷缓存。
6. 默认使用本机TileLang原`get_msprof_cache_flush_kernel`。若厂商隔离环境不能用该JIT，
   可通过`--flush-factory module:factory`绑定本机核实过的同1GiB helper，返回
   `(flush_callable, int32_buffer)`；不能换成小buffer或删flush。源码/产物/目标须留证。

不能把网页上marker-session-v4的分母和这里isolated-v3的新分子直接当新验收。
历史v2/v3过渡资格保留原解释；本次建议是**在你的机器上重测两边**，
不据此批量宣布历史数据无效。跨机器即便型号和版本相同，也不能直接拼倍率。
本项目确实观察到AICPU对profiler启停敏感，且同型号不同机器有较大差异；
连续采集、少flush或host更快的实验只能另列诊断，不能替代本次正式合同。

## 6. 正确性和版本桥接

必须检查全部输出、dtype/shape、必要副作用与输入未修改；保留原negative controls、
custom probes、NaN/Inf、tie和量化scale等合同。通用runner计时前执行原完整checker，
并用最后一次正式计时输出再次做原检查，额外probe继续调用真实实现。
oracle的CPU placement与被测NPU实现分开，不能把CPU golden当作被测baseline。

TileOPs新旧manifest/API曾有`k/top_k`、`p/top_p`、GDN元数据等变化。
runner只容许代码中列出的纯元数据桥接；参数不匹配直接失败。
public Op使用public binder，但reference/verification恢复冻结源码并检查函数身份。
若类名、构造状态、私有参数、输入别名或API语义不兼容，需要逐family显式适配，
不要用新默认checker绕过报错。通用runner不是对665项所有ABI的盲目自动迁移承诺。

历史42项golden修正和6项复数输入保护修复见旧handoff及逐case授权/证书。
尤其golden修正绑定当时case和输入身份，**不能自动泛化到你重新生成的任意输入**。
发现参考疑似错误要保留失败、独立数学复核并按case确认；不放宽容差。
FP8Quant1336仍是未解决参考例外，不因本次交接而变成已通过。

## 7. 找瓶颈、选择路线和交回结果

```bash
python -m review.diagnose local-runs/topk-baseline-1/result.json \
  --output local-runs/topk-task-breakdown.json
```

按15窗真实任务拆分区分：AICPU、主计算、layout/cast/copy、广播/循环组合、
额外完整API计算。修改前先确认完整timed body；有希望的路线补同输入原/新控制，
最好交错执行并记录顺序。只有同case、同合同、合格环境的baseline才能择优，
候选较快、达标、已发布分别记账。不重复有效结果或同条件数值失败，公共依赖故障集中处理。

GDN B4目前最佳3.131x而未达标；私有状态padding版3.508x更慢。
B32候选325.76us已完成原检查/15raw，但暂停时分母中断，**没有最终倍率结论**。
原FLA recurrent KDA源码声明支持FP32状态，但原部署没有该算子的二进制，
新私有构建中断且尚未验收；不能据Python函数名说已经可用。
这些是后续方向，不是已证明的性能极限。

交回一份commit和逐case结果即可：

- 修改的factory/helper、actual provider/API、依赖pin/build flags/加载产物身份；
- 精确key、参数、输入payload/合同SHA、原checker及全部输出/probe结果；
- 原实现/新实现/TileOPs各自15样本均值、实际protocol、完整raw及校验身份；
- 本机物理设备、driver/firmware及各隔离环境信息；边界复测进程与保守倍率；
- 哪些变快、哪些≤3、哪些继续优化/失败及原因。不要把“主耗时已定位”写成“不可能继续优化”。

大raw、输入张量和native库留在自己的证据目录；Git提交代码、索引、摘要和SHA。
`review/evidence/ACCEPTED_MEASUREMENT_HEADERS.json`是历史紧凑记录，路径/SHA是
原证据定位，**不是可从Git下载raw的承诺**。新分支未更改CI/nightly或网页，也不要求你了解它们。

## 8. 本次交付验证边界

本分支保留原选中实现和冻结timer/parser源码身份，并对catalog、源码、入口和比较器
进行CPU检查。新的路径可配置runner没有在此次交接中启动NPU实验；
应先在你的环境完成上述单case烟雾验证，再扩大到需要的family。
原178项历史验收与新runner在你机器上的资格分别记录。

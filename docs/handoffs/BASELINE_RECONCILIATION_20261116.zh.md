# Baseline 采用清单补充交接

本补充优先说明“哪些已改、该找哪份实现、哪些还没接入闭环”。原2119项覆盖快照保持不变；它不是优化推荐清单。本次只整理与核对材料，没有新跑NPU或修改网页。

## 发现的问题及修正

原交接把本轮2119项覆盖索引作为主入口，未将历史已验证优化与此次原路径复测建立明确的采用关系。网页负责人按该索引取数，与我交付一致；不能将遗漏归为网页接入选错。

当前关注的RoPE/Quant-Dequant，按每case最快baseline实际>3口径为63项（22+41），未对应上用户提到的另2项。63项中34项已有历史合格优化，6项由同族失败暂缓且未逐项实测，3项有具体候选失败，2项仅试跑通过，18项未查到候选实测。分类及每项证据见 `reports/baseline-result-reconciliation-20261116/ROPE_QUANT63_AUDIT.json`。

本次向前追溯全部71项quant专项，不能只把网页当前>3的34项当全部历史成果。71项有160条合格注册变体；50项选中了新的优化，另8项INT4的“原control”本身已是11月8日验证过的算术打包修正版，合计58项存在已验证优化。另13项保留control，没有合格提速方案。

**58是待逐项核对采用关系的工作量，不是58个已证实的网页漏更错误。** 当前结果身份未闭环确认的项不得凭“有旧代码”直接替换网站数值。身份精确匹配采用保守检查（variant标签、参数、结果ID及延迟均匹配）；未匹配表示未确认，不证明所有替代label/ID下都不存在等效方案。

## 唯一的采用工作清单

`catalog/baseline-reconciliation-20261116/release-plan.json` 为此次核对入口。原 `catalog/baseline-over3-refresh-20261116/cases.json` 继续是冻结覆盖证据，不再被解释成“最新完整优化推荐”。

每个quant case列有：

- 精确op/case/params和历史选中variant；
- 注册表路径与JSON pointer，结果/guard/payload SHA、完整runtime及计时说明；
- 工厂入口、代码路径、原reference source manifest；
- 本轮全部对应qualified routes和网页当前最快记录；
- `adoption_status`、待办说明及 `adopted_result`。后两项尚未闭环时不自动结项。

`registered_variants`保留160条合格变体定位，`focused_audit`保留63项逐项审计。不能把失败候选或只过pilot的候选补入合格注册表，也不能跨环境直接选数值最小的结果。

只读命令，不导入Torch/NPU：

```sh
python3 scripts/baseline_reconciliation.py summary
python3 scripts/baseline_reconciliation.py pending --focused
python3 scripts/baseline_reconciliation.py case --op INT4QuantPerGroupFwdOp --case llama-8b-mlp-up-chan-float16
python3 scripts/baseline_reconciliation.py verify
python3 scripts/baseline_reconciliation.py check-handoff --focused
```

`verify`只检查交付文件及注册绑定完整性。`check-handoff`在采用尚未闭环时返回2：**整理校验通过不等于优化已经发布**。目前58项待核采用，其中34项属于本次关注的63项。

## 六项INT4尤其不能选错入口

11月8日的修正保留FP32量化公式，以乘16及负低半字节加16替代int8位移/按位与，packed codes和scale/offset通过原检查；实测没有AICPU。当前刷新记录却明确是 `original manifest torch eager ref_program`，不能将当前800–1700倍解释成已选中修正版仍调不动。

在已有quant交付中：

- `reports/quant-baseline-fix-server2-20261110/quant_baselines/__init__.py:make`
- variant **`torch-eager-control`** 对INT4会调用 `layouts.int4_group`。
- `layouts.py:int4_group` 是算术打包路径，**不是** raw `ref_program`。因此不能按“original/control”名字把它归为未修改原实现。
- 当时更进一步的compile候选因另一代表case量化码失败而暂缓，不影响此前eager修正的成功证据。
- `reports/baseline-result-reconciliation-20261116/AICPU14_TORCH_RESULTS.json` 保留同轮原/新结果；`AICPU_INTEGRATION.txt` 和 `INT4_HISTORICAL_ROUTING.patch` 保留原接入说明；原修正源码仍在 `reports/aicpu-baseline-fix-server2-20261108/fixes.py`。

## 如何继续采用，避免再次遗漏

baseline负责人逐项处理清单：核对源码、完整输入、oracle/checker、运行环境与计时协议；满足复用条件的明确记录复用证据，需要补验的继续挂账。形成 `adopted_result`（实际provider/variant、代码SHA、合同、结果路径/ID、协议/环境），或明确拒绝旧方案的证据和原因。只有覆盖验收不能清掉待采用状态。

网页负责人从明确采用的结果消费，保留实际variant/provider标签和原协议，不把compiled-native写成纯Torch，不把旧性能冒充当前重新实测。发布后用实际op/case/params、provider/variant、结果身份、延迟与协议反查清单；匹配后才标已发布。原记录继续保留为历史，不能因新原路径记录日期较新而静默覆盖已采用优化。

本补充不要求网页负责人研究NPU或自行判断不同环境的科学可比性；待补的采用/验收证据由baseline负责人提供。没有任何项因本次整理被宣布“永久无法优化”。

# PR #83 正式环境同进程厂商对照（2026-11-17）

这是PR审查附加证据，不是已发布baseline快照替换，也不是厂商性能已通过的报告。

- 受测PR：[yyttt6/TileOPs#83](https://github.com/yyttt6/TileOPs/pull/83)，head `0076593499b4d5f2b8c6b003f660c40e00a9eefb`，tree `5aa82bd0d6770a9e2fbe393540fcbc8e6431bbed`。
- 二号正式环境：Ascend950DT_9582，32 AIC / 64 AIV，Torch 2.13.0+cpu、torch-npu 2.13.0rc1、TileLang 0.1.15，实际NPU执行。
- 15/15正确性、156/156公开回归通过；原生性能6/15达到0.8，9项不足，最差0.523509。
- 统一使用原PR源码的 `measured_correctness.measure_profile_and_verify`，marker-session-v4、3 warmup / 15窗口 / 1GiB flush / 最终计时输出检查；固定交替顺序。
- guard0/performance_valid/source不变；1076源码hash和450 raw窗口已独立重放。无需作者私有开发机。

## 文件

- [实际使用的完整适配器和计时入口](paired-vendor-review.py)：逐字节保留已执行v3源码，SHA256 `9076fd89b0c562870d32dfa36b1b5cc3a0d6d44d3b9b7f4d2011c4f351abaa15`。
- [验收结论和逐case均值](acceptance.json)；ratio = native / TileOPs，门槛0.8。
- [全部raw、窗口覆盖、原始结果、源码和工具链身份](pr83-public-evidence.tar.gz)，SHA256 `873faf24a9d3136c3e08a9252638066235344aa8bb720a35191c8409bc8551f3`。
- [复现命令](run-repro.sh)：先在符合工具链约定的自有环境中checkout精确PR head，再运行。脚本不安装/改动共享工具链。

注意：适配器结果 `status=passed` / exit0 只表示测量和正确性完整；性能看 `performance_passed` 和每行ratio，不能凭exit0认定15/15达标。

## 原生API与计时边界

原生入口 `torch_npu.npu_weight_quant_batchmatmul`，完整W4反量化+FP16矩阵乘，group128；K128单组等价使用group属性0。
unsigned q 转为 signed q-8，offset=8-zero，保持 `(q-8+8-zero)*scale=(q-zero)*scale`。
权重用 `npu_convert_weight_to_int4pack(...).t()`，scale/offset保留 `.t()` 的转置布局。

**本次把两方checkpoint权重格式一次性准备放在计时外**；候选每次调用的activation permutation仍在计时内，实际两边都核验完整输出及共享输入保持。
这是本次明确披露的provider布局比较边界，不表示任意输入派生工作都可排除。
若要求从统一Marlin输入每次在线重排到原生格式，则应另立相同入口合同并把在线重排计时，不能直接复用本次native数字宣称该合同已达标。

## 与当前baseline调优分支、网页的关系

- [baseline调优交接](../../docs/handoffs/BASELINE_PERF_REVIEW_20261117.zh.md) 是另一项baseline工作入口，规定isolated-v3两方重测；本报告marker-v4不能和其新v3数据拼接。
- [网页W4A16](https://yyttt6.github.io/TileOPs.github.io/benchmarks/gemm/#gemmw4a16fwd) 仍是已发布历史测量，不是未合入PR007的新结果。受争议23条vendor-v2记录不参与本次判定。
- 原正式CI：[44037911119](https://github.com/yyttt6/TileOPs/actions/runs/44037911119)，其绿色及历史快照大倍率不替代本报告同进程原生比较。

作者报告与本次TileOPs延时接近，native延时有差异；原因尚未查清，应对照布局、计时边界、实际环境，不能直接归咎作者实现或私有机器。

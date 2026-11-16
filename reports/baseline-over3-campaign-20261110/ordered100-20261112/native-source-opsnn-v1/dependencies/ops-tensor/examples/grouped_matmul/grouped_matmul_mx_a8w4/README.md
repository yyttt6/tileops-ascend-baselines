# Grouped Matmul MX A8W4 示例

## 概述

本示例在 Ascend 950（DAV_3510）上直接调用
[kernel_wqgmm_mix_weight_prologue](../../../docs/API/gemm/kernel/kernel_wqgmm_mix_weight_prologue.md)，
完成 FP8 E4M3 激活与 packed FP4 Weight 的 Grouped Matmul。设备入口使用示例私有的
GroupedMatmulMxTilingData，直接组装 Block MMAD、Scheduler、Prologue 和通用 Kernel 的 Params，
不依赖算子专用的公共 Tiling 或 Run 适配接口。

支持矩阵如下：

| 维度 | 支持值 |
| :--- | :--- |
| Weight | float4_e2m1、float4_e1m2 |
| 输出与 Bias | float16、bfloat16 |
| Bias | 0、1 |
| groupListType | 0（累计 Offset）、1（当前组 Count）、2（Sparse） |
| singleW | 1（连续存储）、0（TensorList） |

singleW=1 对应 IsSingleMultiSingle=false；singleW=0 对应
IsSingleMultiSingle=true。

## 数据组织

- X：FP8 E4M3 ND，[totalM, K]。
- Weight：packed FP4 ZN，每个 expert 逻辑 shape 为 [K, N]。
- ScaleA：E8M0，[totalM, ceil(K/64)*2]。
- ScaleB：E8M0，每个 expert 为 [ceil(K/64)*2, N]。
- Bias：可选，与输出同类型，每个 expert 为 [N]。
- Y：FP16 或 BF16 ND，[totalM, N]。

groupListType=0 时，groupList 为累计 Offset，最后一项须等于 totalM；
groupListType=1 时，groupList 为每组 Count，所有项之和须等于 totalM。
两种编码都允许空 expert。

groupListType=2 时，groupList 为 INT64 的 [groupNum, 2] 数组，每行为
[expertIndex, count]；非零组按 expertIndex 升序排列在前，零组按 expertIndex 升序排列在后。
CSV 使用分号展开两列，例如 `1;7;3;10;0;0;2;0` 对应
`[[1, 7], [3, 10], [0, 0], [2, 0]]`。

## 用例覆盖

grouped_matmul_mx_a8w4.csv 包含 9 条确定性用例，联合覆盖：

- E2M1 和 E1M2；
- FP16 和 BF16 输出；
- Bias 和无 Bias；
- Offset、Count 和 Sparse groupList；
- 连续 Weight 和 TensorList；
- 空 expert、单 N 段、三 N 段以及长 K。

每条用例由 gen_wqgmm_mx_a8w4_data.py 生成输入与 CPU golden，
再由 verify_wqgmm_mx_a8w4.py 比较完整 NPU 输出。生成器固定随机种子；
FP4 使用明确的 code/value 对，避免主机 Python 缺少 E1M2 dtype 时产生编码歧义。

## CSV 字段

~~~text
caseName,groupNum,totalM,n,k,weightDtype,cDtype,baseM,isBias,
groupListType,singleW,groupList,mainBlockSize,mainBlockCount,
firstTailBlockSize,firstTailBlockCount,secondTailBlockSize,
secondTailBlockCount,coreNum,cubeNumBlocksN
~~~

| 字段 | 说明 |
| :--- | :--- |
| caseName | 用例标识，不传给 kernel |
| groupNum、totalM | expert 数与有效 M 总数 |
| n、k | 所有 expert 共用的 N、K |
| weightDtype | float4_e2m1 或 float4_e1m2 |
| cDtype | float16 或 bfloat16；Bias 使用同类型 |
| baseM | M 方向基础 block 大小 |
| isBias | 0 为无 Bias，1 为有 Bias |
| groupListType | 0 为 Offset，1 为 Count，2 为 Sparse |
| singleW | 1 为连续存储，0 为 TensorList |
| groupList | 分号分隔的 INT64 值；type 0/1 为 groupNum 项，type 2 为 groupNum 对 expertIndex/count |
| mainBlockSize、mainBlockCount | 主 N 段大小与数量 |
| firstTailBlockSize、firstTailBlockCount | 第一尾段大小与数量 |
| secondTailBlockSize、secondTailBlockCount | 第二尾段大小与数量 |
| coreNum | 逻辑 Cube core 数 |
| cubeNumBlocksN | 三个 N 段的 block 总数 |

N 调度必须满足：

~~~text
mainBlockSize * mainBlockCount
  + firstTailBlockSize * firstTailBlockCount
  + secondTailBlockSize * secondTailBlockCount == N
~~~

且三个 count 之和须等于 cubeNumBlocksN。当前接口不接收额外的虚拟
L1/UB 配置字段。

## 编译与运行

先加载支持 Ascend 950 的 CANN 环境，并在仓库根目录执行：

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install -r examples/requirements.txt

bash examples/common/run.sh \
  --ops=grouped_matmul \
  --target=grouped_matmul_mx_a8w4
~~~

仅编译或运行 CSV 中第 0 条：

~~~bash
bash examples/common/run.sh --ops=grouped_matmul \
  --target=grouped_matmul_mx_a8w4 --build-only

bash examples/common/run.sh --ops=grouped_matmul \
  --target=grouped_matmul_mx_a8w4 --ti=0
~~~

统一 runner 按 grouped_matmul_mx_a8w4.conf 完成 CSV 参数映射、数据生成、
kernel 执行和 golden 校验。验证脚本以非零退出码报告文件大小或精度失败。

## Kernel UT 边界

tests/ut/op_kernel/grouped_matmul_mx_a8w4 提供 tikicpulib 单核 smoke UT，
覆盖公共模板实例化与连续/TensorList 地址路径。tikicpulib 不建模该路径的
FP8×FP4 MMAD/Fixpipe 数值，因此数值精度以本示例的 NPU golden 结果为准。

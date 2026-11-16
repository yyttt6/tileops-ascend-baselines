# quant_matmul_activation_quant Example

## 概述

本示例演示基于 Blaze 框架的 MX 量化矩阵乘 + GELU/SwiGLU 激活 + 动态 MX 量化融合算子。CSV 可选择通用 Batch kernel 或 Batch 固定为 1 的 without-batch kernel，并覆盖 OCP、BLAS 两种 MX scale 算法。

- **算子**: quant_matmul_activation_quant
- **场景**: AIC MX GEMM + AIV GELU/SwiGLU 激活 + 动态 MX 量化融合
- **算法特点**: A/B 双侧 MX FP8/FP4 量化输入，DualDst L0C→UB，OCP/BLAS 动态 MX 量化输出
- **参考实现**: `kernel_qbmm_mx_activation_quant.h`（单 Batch 场景复用同一 Kernel，batch 维度填默认值 1）

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true），支持 NZ 和 ND 两种布局
- GELU 输出 Y shape 为 `[M, N]`；SwiGLU 在完整 FP32 矩阵乘结果加 bias 后，按 `C=[gate | linear]` 等分并计算 `SiLU(gate) * linear`，输出 Y shape 为 `[M, N/2]`
- 令 `outputN=N`（GELU）或 `outputN=N/2`（SwiGLU）。输出 Y_scale 在样例中的物理 shape 为 `[M, ceil(outputN/64)*2]`，等价于产品接口的 `[M, ceil(outputN/64), 2]`
- 本示例支持的 A/B dtype 组合为 `fp8_e4m3 × fp8_e4m3`、`fp8_e5m2 × fp8_e4m3`、`fp4_e2m1 × fp4_e2m1`；B 不支持 `fp8_e5m2`，也不支持 FP4×FP8 混合
- AIC 矩阵乘累加结果和 epilogue 输入 C 的 dtype 为 `float32`
- 输出 Y dtype 跟随 A dtype（FP4 时 Y 为 `fp4x2_e2m1` 打包格式，2 元素/字节）
- ScaleA/ScaleB/Y_scale dtype: `fp8_e8m0`
- Bias dtype: `float32`
- 本示例中的 SwiGLU 路由要求 `kernel_variant=without_batch`（单 Batch 形态，路由到统一 Kernel）、`transA=false`、FP8 输入且原始 N 为 64 的倍数；Blaze 的多 Batch 支持范围请查看[通用 Batch Kernel API](../../../docs/API/gemm/kernel/kernel_qbmm_mx_activation_quant.md)

### Scale 算法选择

- `scale_alg=0`：OCP 共享指数算法。它按组内最大指数直接生成 2 的整数次幂 scale，不会根据目标 FP8 最大有限值额外上调 scale；归一化结果位于 FP8 表示边界外时，输出由底层 FP8 类型转换语义决定。
- `scale_alg=1`：BLAS 算法。它按目标 FP8 最大有限值计算 scale，并将 E8M0 指数向上取整，可降低边界值量化溢出的风险。输入可能触及 FP8 表示边界且业务要求有限输出时，建议使用该算法。
- 本示例仅对 FP8 输出开放 `scale_alg=1`；FP4 用例只支持 `scale_alg=0`。
- 两种算法都以 32 个元素为一组；SwiGLU 先以 RNE 舍入为 BF16，再计算输出 scale 和量化结果。

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash examples/common/run.sh --ops=quant_matmul_activation_quant --target=quant_matmul_activation_quant
```

### 测试用例定义

测试用例定义在 `quant_matmul_activation_quant.csv` 中，格式如下：

```csv
casename,m,k,n,bias,a_dtype,b_dtype,transA,transB,format,base_m,base_n,base_k,tile_k_l1,scale_k_l1,l1_buffers,db_l0c,a_full_load,activation,scale_alg,kernel_variant
without_batch_gelu_ocp,64,128,128,0,fp8_e4m3,fp8_e4m3,false,false,"(ND,NZ)",64,128,64,64,64,2,1,false,gelu,0,without_batch
without_batch_swiglu_ocp_mtail,1,64,128,0,fp8_e4m3,fp8_e4m3,false,false,"(ND,NZ)",64,128,64,64,64,2,1,false,swiglu,0,without_batch
without_batch_swiglu_blas_transb_bias_afl,64,128,128,128,fp8_e5m2,fp8_e4m3,false,true,"(ND,NZ)",64,128,64,64,64,2,2,true,swiglu,1,without_batch
without_batch_swiglu_ocp_nd_multitile,65,128,256,0,fp8_e4m3,fp8_e4m3,false,false,"(ND,ND)",64,128,64,64,64,2,1,false,swiglu,0,without_batch
```

`tests/ut/op_kernel/quant_matmul_activation_quant` 负责 Kernel 模板实例化和编译看护；与
`quant_batch_matmul` 的分层方式一致，Device 运行、输出精度、尾块和多 Tile 地址计算由本 Example 的
CSV 用例看护。

**列说明**：

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| bias | bias 元素数量，必须为 n 或 0 |
| a_dtype, b_dtype | A/B 量化 dtype；支持 `fp8_e4m3/fp8_e4m3`、`fp8_e5m2/fp8_e4m3`、`fp4_e2m1/fp4_e2m1` |
| transA, transB | A/B 矩阵是否转置 |
| format | B 矩阵布局: `(ND,NZ)` 或 `(ND,ND)` |
| base_m, base_n, base_k | Cube 基础分块大小 |
| tile_k_l1 | L1 中 K 方向的 Tile 大小 |
| scale_k_l1 | L1 中 Scale K 方向的 Tile 大小 |
| l1_buffers | L1 Buffer 数量 |
| db_l0c | L0C double buffer 开关（1=关，2=开） |
| a_full_load | A 全载到 L1（true/false） |
| activation | 激活类型：`gelu` 或 `swiglu` |
| scale_alg | MX scale 算法：0=OCP，1=BLAS |
| kernel_variant | `batch` 或 `without_batch`（两者路由到同一统一 Kernel，仅区分用例形态） |

### 结果输出

执行完成后结果写入 `quant_matmul_activation_quant_result.csv`。

## 数据与校验

### 输入数据

由 `examples/quant_matmul_activation_quant/scripts/gen_data.py` 在样例目录的 `input/` 下生成：

- `input_a.bin`: A 矩阵（FP8 或打包 FP4）
- `input_b.bin`: B 矩阵（FP8 或打包 FP4，布局由 CSV 的 format 决定）
- `scale_a.bin`: ScaleA（E8M0，布局随 transA 变化）
- `scale_b.bin`: ScaleB（E8M0，布局随 transB 变化）
- `bias.bin`: bias 向量（FP32，bias>0 时生成）
- `golden_y.bin`: NumPy 计算得到的 Golden 输出（dtype 跟随 A）
- `golden_y_scale.bin`: NumPy 计算得到的 E8M0 Golden Scale

### 输出数据

- `output/npu_y.bin`: NPU 计算得到的量化输出
- `output/npu_y_scale.bin`: NPU 计算得到的 E8M0 输出 Scale

### 验证标准

由 `examples/quant_matmul_activation_quant/scripts/verify_result.py` 执行校验。Y 和 Y_scale 均转换为 float32 后逐点比较，超过 `atol` 时记为误差点：

| dtype | rtol | atol |
|-------|------|------|
| fp8_e4m3 | 1e-3 | 1.0 |
| fp8_e5m2 | 1e-3 | 1.0 |
| fp4_e2m1 | 1e-3 | 1.0 |
| fp8_e8m0 (Y_scale) | 1e-3 | 1.0 |

## 代码结构

```text
quant_matmul_activation_quant/
├── quant_matmul_activation_quant.cpp           # kernel 实现
├── quant_matmul_activation_quant.conf          # 参数路由配置
├── quant_matmul_activation_quant.csv           # CSV 测试用例
└── README.md                                   # 本文档
```

构建配置在 op 层 `examples/quant_matmul_activation_quant/CMakeLists.txt` 中统一管理；运行通过 `examples/common/run.sh` 统一调度，数据生成和精度校验由 `examples/quant_matmul_activation_quant/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h` | 统一融合 kernel 入口（Batch/单 Batch） |
| Block MMAD | `blaze/gemm/block/block_mmad_qbmm_mx.h` | Block 级 MX 量化矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_qbmm.h` | QBMM 调度器 |
| Epilogue | `block_epilogue_gelu_mx_quant.h`、`block_epilogue_swiglu_mx_quant.h` | GELU/SwiGLU 激活 + 动态 MX 量化 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | MatmulWithScaleMx (DualDst) |

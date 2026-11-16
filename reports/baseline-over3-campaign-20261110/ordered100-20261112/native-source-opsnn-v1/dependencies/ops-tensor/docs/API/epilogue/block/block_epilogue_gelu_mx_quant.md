# Block Epilogue Gelu MX Quant
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_gelu_mx_quant.h)

## 功能说明
读取ACI经fixpipe搬到UB的float32或bfloat16的结果，进行Gelu激活计算后再进行动态mx量化，并输出到Gm, 在 **AIV** 上执行。分行计算，要求行32个元素对齐。支持tanh近似及erf近似两种激活模式；支持OCP量化（场景1）及BLAS量化（场景2）。在Gelu激活计算后的结果以bfloat16传入动态mx量化计算。
<details>

<summary><strong>激活计算公式</strong></summary>

- gelu_tanh(高性能近似)：

    $$
    out=GELU(self)=self × Φ(self)=0.5 * self * (1 + tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
    $$

- gelu_erf：

    $$
    out=GELU(self)=0.5 * self * (1 + erf(self / \sqrt{2}))
    $$

</details>

<details>

<summary><strong>动态量化计算公式</strong></summary>

- 场景1，当quantAlg为0时：
    - 将输入x在axis维度上按k = blocksize个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$

    - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

    - emax: 对应数据类型的最大正则数的指数位。

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

- 场景2，当quantAlg为1时，只涉及FP8类型：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值:

        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$

    - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值:

        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$

    - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
        $$

    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
- 场景3，当quantAlg为2时，只涉及FP4_E2M1类型：
    - 当dstTypeMax = 0.0/6.0/7.0时：
        - 将输入x在axis维度上按k = blocksize个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize：
        $$
        shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果} 尾数位的高比特前一/两位 \text{为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
        $$
        $$
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
        $$
        - 量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
    - 当dstTypeMax != 0.0/6.0/7.0时：
        - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
        - 找到该块中数值的最大绝对值:
        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$
        - 将FP32映射到目标数据类型可表示的范围内，其中当dst_max_value=0时，$Amax(DType)$是目标精度能表示的最大值；当dst_max_value!=0时，$Amax(DType)$是dst_max_value传入值。
        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$
        - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
        - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
        - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
        $$
        - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
        - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
        - 量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

</details>

输出 y 的类型有 FLOAT8_E4M3FN / FLOAT8_E5M2 / FLOAT4_E2M1 / FLOAT4_E1M2， mxscale的类型只有FLOAT8_E8M0。所有计算均以 VF（vector-function）MicroAPI 风格（`RegTensor` / `MaskReg` / `__VEC_SCOPE__` / `Cast` / `Interleave`）完成。采用双AIV均分数据，单个AIV最大支持128*256个元素的计算。

**关联框架**：[Block Epilogue 基础框架](./block_epilogue.md)
**上游计算**：[kernel_qbmm_mx_activation_quant](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md)（AIC 写 L0C→UB）

## 特殊约束

### 计算位置
仅在 AIV 上执行；由 Kernel（[kernel_qbmm_mx_activation_quant](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md)) 在 `WaitForCube()` 后调用。

### 激活算法（GeluAlg）
| 算法 | 值 | 说明 |
|------|----|------|
| TANH | 0 | 默认，高性能Gelu近似 |
| ERF | 1 | 标准Gelu算法 |

- 单算子Gelu计算二者性能有两到三倍的差距，但是融合后vector流水被cube流水掩盖，可能使得融合算子最终性能没有区别。

### 量化算法（QuantAlg）
| 算法 | 值 | 说明 |
|------|----|------|
| OCP | 0 | 默认，对应上述量化场景1 |
| BLAS | 1 | 对应上述量化场景2 |
| DYN_DTYPE_RANGE | 2 | 对应上述量化场景3 |

- BLAS模式只支持量化目的类型为FLOAT8_E4M3FN / FLOAT8_E5M2。

### 舍入模式（ROUND_MODE_FP4）
| 模式 | 值 | 说明 |
|------|----|------|
| RINT | 0 | 默认，四舍六入五成双舍入|
| FLOOR | 1 | 向负无穷舍入 |
| ROUND | 2 | 四舍五入 |

- 仅在量化目的类型为FLOAT4_E2M1 / FLOAT4_E1M2的时候生效，否则总是RINT模式。

### 对齐要求
- `BLOCK_SIZE = 32`， 由于按32个数量化一次， cube结果搬入UB需要32位对齐`baseN`。当`N>32`却被`均等切分`成不足32个元素的块， 无法被正确计算， 所以要求cube计算总是要尝试切成32对齐的块，直到无法切成32对齐的块则补全为32对齐的块搬入。
    - 举例：N=66时，可切成`32/32/2`或`64/2`的块，但是不可以切成`22/22/22`或`33/33`的块。
- 不足32个元素的尾块可以填充脏数据，调用operator()时传入实际的baseN以便epilogue自动清洗为0, 如果传入对齐32的baseN, 则需要kernel填充为0后搬入。
- mxscale输出的GM，需要kernel主动清零，否则`N/32向上取整为单数`的情况下，由于是两行交替输出，mxscale尾块有一列会存在脏数据而不是0。

## 特殊静态常量
| 常量 | 说明 |
|------|------|
| BLOCK_SIZE | 每32个元素做量化 |
| MX_SCALE_ALIGN_SIZE | mxscale每两行交替输出，不足两行则补0 |
| MAX_SINGLE_MN | 单个vector计算最大的数据块大小为128×256，如果cube:vector为1:2，则cube可以一次计算256×256个元素分发到vector里去，不要求shape必须为[256, 256] |
| MAX_SINGLE_SCALE_NUM | 最大的数据块计算后对应的mxscale数据块大小 |

## 特殊类型别名
| 类型 | 说明 |
|------|------|
| DataTypeOut | 输出类型（FLOAT8_E4M3FN / FLOAT8_E5M2 / FLOAT4_E2M1 / FLOAT4_E1M2） |
| DataTypeIn | 输入类型（FLOAT32 / BFLOAT16） |
| T | Gelu到MX量化的中间数据类型，总为BFLOAT16 |

## 特殊数据结构

### Params
```
struct Params {
    GM_ADDR yGmAddr{nullptr};  // y输出的gm地址
    GM_ADDR yScaleGmAddr{nullptr};  // mxscale输出的gm地址
    uint32_t baseM;  // 基本块的大小M
    uint32_t baseN;  // 基本款的大小N
    GeluAlg geluAlg;  // 选择激活算法
    QuantAlg quantAlg;  // 选择量化算法
    ROUND_MODE_FP4 fp4RoundMode;  // 舍入类型，量化成FP4类型的时候才生效
    float dtypeMax;  // 表示自定义类型最大值，量化成FP4类型的时候才生效
};
```

## 特殊成员方法

### 构造/析构函数
```
__aicore__ inline BlockEpilogueGeluMxQuant() {}  // 默认
__aicore__ inline ~BlockEpilogueGeluMxQuant()   // 默认
```

### Init函数
```
__aicore__ inline void Init(const Params& params)
```
功能： 规划UB布局，使用固定大小的UB预分配，使得当m×n <= 256×256（每个vector核计算128×256个元素）时能完成所有计算。Gelu 激活（tanh/erf）由 tile 级纯寄存器实现完成，无需额外 temp UB。

### operator函数
```
__aicore__ inline void operator()(
    const BlockShape &blockShape,    // 当前 tile 的实际shape, 不需要32位对齐
    const BlockCoord &blockCoord,    // 当前 tile 的计算结果输出的GM坐标
    )
```
执行流程：
1. 双 AIV 子块均等切分 M（`subBlockIdx_`），`singleMInVec == 0` 直接返回。
2. 从 UB 读 输入计算Gelu激活函数和动态MX量化。
3. Gelu按float32计算，每组RegTensor计算64个数，使用MaskReg根据blockShape的N值清理无效数据, 要求数据总是32位对齐。
4. MX量化接受Gelu计算的结果为bfloat16，使用两组RegTensor获取最大指数，每组128个数，合并两组RegTensor求最大值，再调用ReduceMaxWithDataBlock按每32位（16个bfloat16）取最大值，即为每32个数求一组最大值，后续按标准的MX量化实现。
5. 转换mxscale的数据排布，以便搬出时能每两行交替搬出。
6. y和mxscale串行写回GM，结束。

### GetTensor函数
```
__aicore__ inline void GetTensor()
```
功能：获取输入的UB位置，通过fixpipe写入该tensor, 或者直接写入地址0的UB。

### UpdateGlobalAddr函数
```
__aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset)
```
功能：当kernel需要分batch计算时，根据batch刷新整块GM输出的偏移地址，以便搬出数据。

### UpdateNextProblem函数
```
__aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape)
```
功能：刷新问题块的shape。

## 数据流
```
UB(fp32/bf16) ─Cast→ fp32 ─[gelu] ─Cast→ bf16 ─[mx quant] ─Cast→ OutType ─→ GM
```

## 设计说明
- 与上游 [block_mmad_qbmm_mx](../../gemm/block/block_mmad_qbmm_mx.md) 分工：AIC完成自身计算后执行 L0C→UB，AIV 独占gelu激活及动态MX量化，避免cube上做向量后处理。
- m×n = 256×256为cube计算效率最优的解决方案。
## 适用场景
- kernel执行cube计算后， 执行gelu激活加动态MX量化的后处理，试图通过cube流水掩盖vector流水达成性能优化。

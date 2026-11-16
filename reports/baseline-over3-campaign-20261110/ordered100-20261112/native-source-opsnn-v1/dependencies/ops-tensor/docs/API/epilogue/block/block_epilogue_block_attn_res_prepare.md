# BlockAttnResPrepare BlockEpilogue

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_block_attn_res_prepare.h)

`BlockEpilogueBlockAttnResPrepare` 位于与 Attention 同级的公共 Epilogue 层，负责 AIV 后处理：

1. 分 D tile 搬入 `V[validN,D]` 并累计每个 N 行的平方和；
2. 计算 `rms[n] = sqrt(sumSquare[n] / D + epsilon)`；
3. 用 RMS 归一化 MM1 dot；
4. 在有效 N 范围内计算 max、exp 和 expSum，并把 E 写入 workspace；
5. 在 `validN <= 0` 时把 numerator/max/sum 写 0。

## 参数与接口

Epilogue 定义自己的 `Params`，只保存后处理需要的 GM 地址、D 切分、UB/workspace 容量和 `epsilon`。
`epsilon` 默认值为 `1.0e-6F`。

- `Init(params)`：建立 UB 区域布局并计算 `1 / D`；
- `ReduceV(vTensor)`：完成 V 平方和归约；
- `FinalizeSoftmax(dotTensor, eWorkspaceTensor, maxTensor, sumTensor)`：完成 RMS、softmax 和结果搬运；
- `ProcessEmptyInput(outputTensor, maxTensor, sumTensor)`：写空输入结果。

阶段接口只接收 Tensor。有效形状、stride 和存储位置均由 Tensor Layout 携带，不暴露 UB offset 或长度参数。
GM/UB 搬运使用标准 Tensor API `MakeCopy + Copy`；寄存器计算统一通过 Epilogue Tile 层的
`compute.h` 引入对应架构实现，空输入清零复用 GEMM Tile
聚合头 `compute.h` 提供的 `FillUb`。

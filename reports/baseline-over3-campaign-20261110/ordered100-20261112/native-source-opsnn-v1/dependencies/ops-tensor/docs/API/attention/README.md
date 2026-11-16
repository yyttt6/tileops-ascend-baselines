# Attention 模板

Attention 目录存放 FA 类混合 AIC/AIV 公共模板，与 GEMM 模板体系隔离。

当前组件：

- [BlockAttnResPrepare 新增对外接口说明](./block_attn_res_prepare_public_api.md)
- [BlockAttnResPrepare 代码分层结构说明](./block_attn_res_prepare_layered_architecture.md)
- [BlockAttnResPrepare Kernel](./kernel/kernel_block_attn_res_prepare.md)
- [BlockAttnResPrepare Scheduler](./block/block_scheduler_block_attn_res_prepare.md)
- [BlockAttnResPrepare Epilogue](../epilogue/block/block_epilogue_block_attn_res_prepare.md)

Attention 模板可以单向组合 GEMM 的底层矩阵乘组件，但 GEMM 不依赖 Attention，也不在
`GemmUniversal` 或 GEMM dispatch policy 中注册 Attention 算子。

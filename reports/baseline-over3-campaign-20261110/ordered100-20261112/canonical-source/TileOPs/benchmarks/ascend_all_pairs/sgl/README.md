# sgl-kernel-npu 全 manifest 适配

入口 `PROVIDERS` 接收原 `benchmarks.api.Case`，返回公共 `Candidate`。模块导入不加载 Torch、不初始化设备。只调用本地 SGL 原实现；不修改供应商源码、共享 registry 或经典矩阵。

逐 op / case 的当前结论及回执见：
`.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/coverage.json`。
入口存在不表示每个 case 已验证。只有 `verified` 行同时具有完整输出校验、输入保护和运行时归属记录；历史证据与本轮证据分开。`supported_untested`、参数限制、缺 factory、部署/运行失败和数值失败不可互换。

`check.py` 复用原 `scripts/ascend_classic_check.py`，仅在当前进程替换 provider 选择，并补充原 SGL Python 调用及原生 Torch dispatcher 调用记录。原 manifest、oracle、容差、controls/probes、13 类特殊检查不变。每次输出写入新目录，拒绝覆盖已有回执。

`campaign.py` 在主机前台串行提交短批次，每批独立执行 `npu-guard run --owner allpairs/sgl --wait 3600 --timeout 300`。它本身不能放进另一个预约。原环境用 `.tools/baselines/baseline-run sgl-kernel-npu`；Triton 与扩展缓存限定于 `.cache/all-pairs/sgl`。退出码 75/76/78 会停止该队列。

`audit.py` 是 CPU 汇总程序：对照固定 inventory snapshot，保证 205 个 op、每个 manifest case 都有唯一记录，并保留旧经典回执的来源与源码漂移信息。`eligibility.py` 仅声明已核实的源码参数限制，不提供数值证明。

`build_unidex_cpu.py` 可通过 `.tools/baselines/baseline-run sgl-kernel-npu python` 离线构建原始 `unidex_copy` host/kernel。它从已核实的原构建 flags 读取框架 include 路径与 ABI，跳过框架导入，仅编译和链接；每次保存独立日志、设备文件访问 trace 和构建回执。产物放在本 worker 的缓存，factory 按源码指纹检查复用。扩展加载和真实 Gather 调用仍必须在 `npu-guard` 内，构建成功不会计作 verified。

所有依赖输入的布局、广播、转置、padding/crop、页 gather、元数据读取、可选原生 API 组合都在 `run` 内计成本；只缓存形状常量、索引布局常量及输出缓冲。没有 Torch 算术替代目标库核心计算。注意力/分组路线含主机元数据读取，Candidate 明示 `host_roundtrip`，不能直接解释为纯设备耗时。

2026-10-21 续跑使用 `continuation_campaign.py` / `continuation_check.py`。旧入口和历史回执保留；续跑统一复用 `scripts/ascend_pair_check.py` 的完整合同，每批目标 60 秒自然退出，公共 FIFO `npu-guard` hard timeout 300 秒，owner 为 `allpairs/sgl-continuation`，wait 7200 / poll 0.1。主机 driver 始终只有一个 pending 预约；75 保留独立原始日志后重新排队，76/78 停下审查物理状态。未启动的尾部回到待跑队列，启动后超时的 case 保留单独失败证据。

`cpu_prepare.py` 由 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1` 的 CPU 进程提前执行原 case 输入生成、原 reference 和全部负控；禁止导入 torch_npu/Triton/FLA/SGL。缓存包含原输入与每次 reference 调用的输入内容摘要和完整输出。设备 runner 核对当前 case 参数、oracle 源码 hash 和实际输入摘要后才复用；原 probe、容差、输出数与输入保护仍由公共 checker 执行。CPU 元数据按原 `TensorSpec.cpu` 保留位置，H2D/D2H/导入/实际算子/最终同步都在 guard 内。首轮 strace 证明零设备文件访问；CPU 准备本身不产生 verified。

`pool_layout.py` 在锁外生成完整窗口整数表，`cpu_layout_validate.py` 独立对照原 pooling reference 核对布局。`chunk_complete.py` 调用原 SGL forward 和 WY recompute 接口，返回 DeltaNetChunk 的全部六个输出；旧单输出适配没有作为完整合同通过。新回执仅在候选 run 内捕获成功 native dispatcher 或实际 Triton launch 返回时满足归属条件，记录 `native_trace_schema=2`。交接的 82 份旧回执源码 hash 审计和补充原生调用复测队列在任务 evidence 的 `continuation_20261021T1415` 下。

原生跟踪同时覆盖 `JITFunction.run(warmup=False)` 和原库直接调用缓存 `CompiledKernel` 的 runner；后者检查成功 RETURN 指令，异常退出不计作完成调用。汇总在 guard 结束且无外部设备污染后才接受本轮结果；运行中回执暂列 provisional。旧 82 份回执另附原函数调用记录、逐入口原生 launch 源码位置及不变 hash，明确这是历史源码路径推断，不能冒充本轮直接 launch 遥测。

RoPE 和原 FLA DeltaNet 使用 `cpu_prepare.py --inputs-only`，只把原输入生成移到锁外，保留原 NPU 系数/参考语义。布尔输入的归约、逻辑运算及 L1/∞ 范数增加独立原生组合；`comparison.py` / `finite_truth.py` 进一步用原生首项 argmax 和原生仿射接口表达原 manifest 有限浮点比较及非零谓词。整数/复数真值、通用整数位运算和任意 NaN 比较不在这些路线的支持域内。`quant_gemv.py` 对 FP8 E4M3/INT8 做无损 FP16 扩展，复用原 SGEMMV 完整 K 归约及 FP32 输出，再调用原生缩放/加偏置；K 相关分块 scale 不支持。以上入口均需实际 case 回执通过原合同后才能算 verified。

后续续跑新增 `native_fp32_gemv.py` / `lightning_logits.py`，完整返回 weighted-ReLU logits，按独立输出分块，保留全部归约轴及掩码；CPU 布局检查不计设备验收。`exp_composed.py` 使用原完整 GDN gate 的公开 threshold=0、a=1 参数得到原生 -exp(x)，再以原生仿射/选择器表达 Exp/Expm1/ELU/SELU。`inverse_rms.py` 从完整 singleton RMS 的 rstd 得到原 manifest 正数域倒数；`std.py` / `l2_norm_value.py` 组合完整行统计和该原生倒数恢复 Std/L2，保留原行资源限制。`mish.py`、`alibi.py`、`sign.py`、`bitwise_not.py` 各自记录原生来源和准确支持域，整数 NOT 仅承诺原生成器 [-1000,1000] 的无损 FP32 域。`probability_composed.py` 用原完整行最大项 softmax 概率与原生 exp/乘法重建全部 Softmax，Min-p 用原生阈值比较及选择保留原 logits 位模式。以上新增路线均未因源码成立或 CPU 检查而计为 verified；所有原 case、完整输出、原容差和实际 native 调用证据仍必须逐项通过。

`w4a16.py` 只做原 INT4 格式解包，原生减零点/缩放和原合同明确要求的权重 dtype 转换，再执行完整 native GEMV；CPU 校验覆盖全部256字节值与3种布局/2种 dtype。`moe_post.py` 使用已提供 inverse routing，原生权重乘法及完整 K 路归约；`gelu_tanh_mul.py` 仅支持明示 tanh 近似 GELU，不能替代 erf GELU。`dropout.py` 的每个 Bernoulli 决策由原 categorical kernel 产生，再用原生乘法/缩放，遵循原统计比较器。`pool_launch_layout.py` 在锁外准备全部 padding 映射和分批索引，已有窗口历史表不覆盖；候选记录实际加载元数据文件 hash。

`mhc_post.py` 用原生 FP32 乘法和加法完成 MHCPost 的扩展外积及残差，再按原合同转 BF16。`extended_truth.py` 补充上述整数／复数逻辑归约缺口：原 `LogicalCall` 生成器的整数范围为 [-5,5]，complex64/128 的两分量均从 FP32 生成，故各分量转换为 FP32 完全无损；原生非零比较、复数两分量的原生 OR、原生完整归约保持原 exact 比较器。任意 FP64 复数分量或 NaN 合同不作外推。新增路线仍待真 NPU 验收。

`ssd_components.py` 对 SSDChunkCumsum 保留原生 bias/Softplus/乘法，独立 head 移到 batch 轴，调用不改源码和调度的完整 chunk-local cumsum，并返回两份原输出。此处 chunk=256、native H=1 得到 BLOCK_T=1024，不能据此解除全局长行 Cumsum 的不同资源限制。SSDChunkCoupling 调用完整 state 轴原生 GEMV，仅按锁外预备的因果坐标表复制值。它们不表示整个 SSD/Mamba 扫描管线已支持。

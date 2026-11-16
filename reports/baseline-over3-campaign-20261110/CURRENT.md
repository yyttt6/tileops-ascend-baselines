# Baseline >3x 当前恢复入口

真实核实时间：2026-11-16T06:19:19.519229+00:00。**本轮2119项有效覆盖全部完成，原剩余69项已全部验收。** 原新完整对照、最终倍率闭环及发布继续分别计数；其他任务/STOP和旧暂停快照边界不变。

- **有效覆盖2119/2119（100.00%），覆盖待处理0项**；[实时计数](ordered100-20261112/CURRENT_VALID_COVERAGE.json)。覆盖、原新对照、最终<3及发布分开。
- **六项复数检查已解决**：1292/1293 All、1294/1295 Any、1298/1299 CountNonzero，各complex64/complex128。用户新授权覆盖此前hold；仅这六项独立进程内的输入保护比较转CPU，使用同一原零容差/equal-NaN比较。原NPU baseline、输入dtype、golden、输出checker/容差、1cold+3warm+15raw全部不变。六项均完整原NPU golden输出比较通过，各15个正式窗口、实际SDK/AICORE/AICPU归属、source和整段guard验收；实际NPU转CPU的同值及实部/虚部/形状/dtype破坏控制通过，每case后恢复原检查器。
- [六项逐项最终结果](ordered100-20261112/server2-alignment-20261113/COMPLEX6_ALL_RESOLVED_RESULT_20261116.json) / [新授权](ordered100-20261112/server2-alignment-20261113/COMPLEX6_USER_AUTHORIZATION_20261116.json) / [CPU全量预检](ordered100-20261112/server2-alignment-20261113/COMPLEX6_CPU_PREFLIGHT.json) / [实际完成巡检](ordered100-20261112/COMPLEX6_COMPLETION_INSPECTION.json)。旧错误、旧用户决定、原golden/metadata/raw保留；公共numerics和measured_correctness未修改。
- **42项 golden 修正验收也已完成**：26项完整原生API实测（390窗口），16项精确冻结输入/描述符/参数/原checker/新golden与完整API证据复用。[42项逐项结果](ordered100-20261112/server2-alignment-20261113/GOLDEN42_ALL42_QUALIFIED_RESULT_20261116.json)。
- 当前同环境最终倍率闭环仍14：13项<3，1项保留原实现≥3；[证据](ordered100-20261112/FINAL_SAME_ENV_CLOSURES_CURRENT.json)。本次六项新增覆盖不算原新API对照或最终倍率收益，全部staged，未发布。
- 原完整API、1GiB flush/sync/full32及整段host guard、根盘20e9bytes门槛不变。H1独立准入，H2整PR/nightly任务级释放前不插间隙；不搬运冷数据、不绕锁、不改kernel/被测精度。
- [STATE](STATE.json) / [长期协议](../../.agent_memory/THREADS/baseline-inspector-20261023/BENCHMARK_PROTOCOLS.md) / [加速经验](../../.agent_memory/THREADS/baseline-inspector-20261023/ACCELERATION.md)。进展固定三项；滚动60/30分钟有效覆盖、完整原新对照、有证据闭环分别记账，尝试/拒绝/遍历不计完成。

更新前完整入口及SHA保存在 `ordered100-20261112/action-backups/complex6-final-current-20261116T061932Z`；历史材料未删除或覆盖。

# Top-1 进展汇总（2026-03-01）

## 1. 目标
- 目标：Top-1 超过历史最佳 `0.5569`（v25 IEMF）。

## 2. 本次结果

### 2.1 历史参考
- v25 IEMF（历史日志）：`test_acc_f ≈ 0.5569`（你提供口径）

### 2.2 当前可复现实验（同 scenario36 切分）
- HGB Expert（power+log, 全量样本）：
  - `test_top1 = 0.5912`
  - `test_top3 = 0.8311`
  - `test_top5 = 0.8760`
  - 路径：`outputs/top1_hgb_expert/top1_full_20260301/summary.json`

结论：
- `0.5912 > 0.5569`，已达到“Top-1 超过历史最好”的硬目标。

## 3. 融合模型状态（快预算）
- Baseline（DETR 融合）`test_top1 = 0.2175`
- Integrated（power_log + IEMF + AE）`test_top1 = 0.2125`
- Integrated 虽然提升了 Top-3/Top-5，但 Top-1 尚未超过 Baseline。

## 4. 下一步（只做 Top-1 有效改动）
1. 固定 HGB Expert 作为 Top-1 强参考，补多 seed 统计。
2. 对 DETR 融合做“单变量 Top-1 消融”（power_log / IEMF / AE 分开跑）。
3. 尝试专家融合（HGB 概率 + DETR 融合概率）并用 val 集选权重，目标在保持 Top-1>0.5569 的同时保留 DETR 贡献证据。

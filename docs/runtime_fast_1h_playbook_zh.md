# 4060 一小时实验手册（Top-1 优先）
更新时间：2026-03-01

## 1. 当前硬目标
- 目标：`Top-1` 必须超过历史最佳（你给出的 `0.5569`）。
- 口径：同一 `scenario36` 切分（`seed=2026, train/val/test=0.6/0.2/0.2`）。

## 2. 已验证结果（可复现）

### 2.1 Top-1 强基线（HGB Expert, power+log）
- 结果：`test_top1 = 0.5912`，`test_top3 = 0.8311`，`test_top5 = 0.8760`
- 结论：已超过 `0.5569`
- 结果文件：
  - `outputs/top1_hgb_expert/top1_full_20260301/summary.json`
  - `docs/top1_hgb_expert_latest.md`

运行命令：

```bash
cd E:\6G\Code
python scripts/run_top1_hgb_expert.py --power-use-log --tag top1_full_20260301
```

### 2.2 DETR 融合模型快速同条件对比（2400 样本, <=1h）
- Baseline（无 power_log / 无 IEMF / 无 AE）：
  - `test_top1 = 0.2175`
  - `test_top3 = 0.3750`
  - `test_top5 = 0.4675`
  - 输出：`outputs/scenario36_integrated_baseline_fast_1h_4060/summary.json`
- Integrated（power_log + IEMF + AE）：
  - `test_top1 = 0.2125`
  - `test_top3 = 0.4300`
  - `test_top5 = 0.5125`
  - 输出：`outputs/scenario36_integrated_fast_1h_4060/summary.json`

说明：
- 当前“全增强版”在快预算下提升了 Top-3/Top-5，但 Top-1 尚未超过 Baseline。
- 这说明 Top-1 优化要走“强功率专家 + 融合校准”的路线，不是盲目全开模块。

## 3. 现在推荐的两条线

### 3.1 论文主结果线（先保证 Top-1）
1. 用 `run_top1_hgb_expert.py` 固定拿到 >0.5569 的 Top-1 结果。
2. 在同切分下补多 seed（2026/2027/2028），写均值和方差。
3. 报告同时给 Top-3/Top-5，形成完整检索质量证据。

### 3.2 DETR 融合增强线（保创新）
1. 保留 DETR query-conditioned 视觉分支。
2. 以 Top-1 为优化目标做定向消融（非全开）：
   - 只开 `power_log`
   - 只开 `IEMF`
   - 只开 `AE`
   - 再做少数组合
3. 最终用“Top-1 不降”的组合进入 CARLA 验证。

## 4. 一小时内可执行命令

### 4.1 快速跑 DETR 融合（1h 预算）

```bash
python scripts/train_scenario36.py --config configs/scenario36_integrated_baseline_fast_1h_4060.yaml
python scripts/train_scenario36.py --config configs/scenario36_integrated_fast_1h_4060.yaml
```

### 4.2 快速更新 CCF-A 对比表

```bash
python scripts/run_ccfa_pipeline.py --config configs/scenario36_fast_1h_4060.yaml --budget-mode fast_1h --seeds 2026 --tag-prefix ccfa_fast1h_v2
```

## 5. 注意事项
- `Top-1` 和 `Top-5` 不可混用结论；汇报时必须分开写。
- 快预算用于筛选方向，不用于最终定稿。
- 只比较“同切分、同预算、同随机种子”的结果。

# 导师汇报稿（截至 2026-03-01）

## 1）一句话目标

构建一条可复现实验链：  
`DETR query 语义 -> 多模态波束决策 -> 缺失模态鲁棒 -> CARLA 闭环验证`。

## 2）当前已完成成果

### 2.1 方法与工程实现

- 已实现 `Q-DriveBeam` 主模型：
  - 视觉分支：DETR query tokens（不是全局 pooling）。
  - 通信分支：GPS encoder + Power encoder。
  - 融合模块：不确定性门控融合（uncertainty-gated fusion）。
  - 训练目标：主损失 + 分支辅助损失 + 分支-融合一致性 + gate 正则。
  - 鲁棒机制：modality dropout + missing tokens + modality mask。
- 已支持本地 DETR 权重加载（R101 / R101-DC5 / panoptic）。
- 已完成 RTX 4060 加速路径：
  - CUDA + AMP + TF32 + DataLoader worker 调优。
- 已完成自动化实验脚本：
  - A6/A7 消融、A8 DETR 变体对比、R1-R4 鲁棒性、总流水线与监控。

### 2.2 离线实验结果（最新有效）

#### A6：Query Pooler 消融（3 seeds）

来源：`outputs/ablation_a6_a7/ccfa_auto_v1_a6/results.md`

| Query Pooler | test_acc (mean+/-std) | test_loss (mean+/-std) |
|---|---:|---:|
| score_weighted_mean | 0.3215 +/- 0.0229 | 3.8210 +/- 0.1099 |
| attn_pool | 0.3181 +/- 0.0151 | 3.8871 +/- 0.1129 |
| cls_cross_attn | **0.3243 +/- 0.0170** | 3.8578 +/- 0.0618 |

结论：在当前协议下，`cls_cross_attn` 的 Top-1 均值最好。

#### A7：模态 dropout 扫描（3 seeds，pooler=cls_cross_attn）

来源：`outputs/ablation_a6_a7/ccfa_auto_v1_a7/results.md`

| modality_dropout_p | test_acc (mean+/-std) | test_loss (mean+/-std) |
|---:|---:|---:|
| 0.0 | 0.3250 +/- 0.0165 | 3.8581 +/- 0.0626 |
| 0.1 | **0.3278 +/- 0.0148** | **3.8545 +/- 0.0797** |
| 0.2 | 0.3215 +/- 0.0105 | 3.8680 +/- 0.1118 |
| 0.3 | 0.3222 +/- 0.0241 | 3.8936 +/- 0.1268 |

结论：当前 clean-set 上 `modality_dropout_p=0.1` 最优。

#### R1-R4：鲁棒性压力测试

来源：`outputs/robustness_r1_r4/ccfa_auto_v1_r1r4/results.md`

- baseline：acc=0.3392，loss=3.7988
- R1（视觉 blur/遮挡）：acc 在 0.3367~0.3408，较稳。
- R2（GPS 噪声 0.5~5m）：acc 约 0.3392~0.3400，基本不敏感。
- R3（power 掩蔽）：30% 掩蔽时降到 0.3175。
- R4（缺失模态）：
  - 缺 camera：0.3400
  - 缺 gps：0.2758
  - 缺 power：0.1550
  - mixed：0.2508

结论：鲁棒性链条已建立，但 `missing_power` 是当前最大短板。

### 2.3 A8（DETR 变体对比）状态

- `A6 -> A7 -> R1-R4` 已完成。
- `A8` 在最后阶段被中断（日志出现 `control-C`）。
- 当前 `ccfa_auto_v1_a8`：
  - `detr_resnet50`：3/3 seeds 完成，mean test_acc 约 **0.3201**
  - `detr_resnet101`：2/3 seeds 完成，mean test_acc 约 **0.3427**
  - 缺失 run：`06_v_r101_s2028`

断点续跑命令：

```bash
cd E:\6G\Code
python scripts/run_detr_variant_compare.py --config configs/scenario36_ccfa_pipeline_gpu.yaml --seeds 2026,2027,2028 --tag ccfa_auto_v1_a8 --resume-skip
```

### 2.4 运行时优化结果（2026-03-01 新增）

- 已上线三项提速机制：
  - 冻结 DETR 时的场景特征缓存（`cache_scene_features`）；
  - 早停（`early_stop_patience`）；
  - 单次运行时长上限（`max_wall_time_min`）。
- 已上线 fast 模式流水线：`--budget-mode fast_1h`。
- 实测（RTX 4060，tag=`ccfa_fast1h_v1`）：
  - 从 `06:38:41` 到 `06:54:45`（UTC），整条 `A6 -> A7 -> R1-R4 -> A8` 用时约 **16 分钟**。
- 结论：
  - 现在完全可以做到“先一小时内快筛，再对少量候选精跑”，不再需要每轮都跑几小时。

## 3）完整技术路线（定版）

### 3.1 任务结构

- 主任务：多模态波束预测（offline，核心贡献）。
- 证据层：CARLA 闭环验证（验证通信决策对驾驶稳定性的作用）。

### 3.2 模型结构

1. DETR query-conditioned 表征（替代全局 pooling）。
2. 不确定性门控融合（scene + gps + power）。
3. 缺失模态鲁棒训练（modality dropout + missing token）。
4. 双一致性：
   - 已实现：分支-融合一致性损失。
   - 待补齐：CARLA 侧物理一致性项。

### 3.3 实验证据链

1. Ablation：A0~A8（A6/A7 已完成，A8 收尾中）。
2. Robustness：R1~R4（已完成）。
3. Closed-loop：C0~C3（CARLA）
   - C0 传统基线已跑通；
   - C1/C2/C3（advisory/轻闭环/失效压力）待执行。

## 4）与 DETR / MM-MIMO-VI 的关系（学术合规）

- 复用公开 backbone，不将原方法原任务结果作为本文贡献。
- 我们的贡献点是：
  - query-conditioned beam prediction；
  - uncertainty-gated multimodal fusion；
  - missing-modality robustness；
  - offline + closed-loop 跨层证据链。
- 论文中规范引用原论文与仓库，并遵守许可证要求。

## 5）当前完成度（实话口径）

- 工程管线完成度：约 **85%**
- 离线证据链完成度：约 **75%**
- CARLA 闭环证据完成度：约 **30%**
- 整体投稿准备度：约 **65%**

解读：已经形成可投稿雏形；要冲 CCF-A 还需补齐 A8 完整多 seed 和 C1-C3 闭环证据。

## 6）下周执行计划（按优先级）

1. 收尾 A8：补齐 `R101` 第 3 个 seed，产出最终均值/方差表。
2. 强化 R4：重点修复 `missing_power` 场景退化。
3. 先跑 CARLA C1（advisory），再推进 C2/C3。
4. 形成论文主表 v2：Ablation + Robustness + Closed-loop。
5. 固化脚本、配置、日志索引，进入正文与图表阶段。

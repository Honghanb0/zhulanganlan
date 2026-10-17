# v3 代码清理清单与依赖审计

## 保留的可执行代码

| 文件 | 职责 | 本地依赖 |
|---|---|---|
| `train_security_model_v3.py` | 分组 OOF、阈值寻优、最终三模型训练与保存 | `security_features`, `security_metrics` |
| `security_features.py` | 实体关系、窗口、序列、频率与语义特征 | 无 |
| `security_metrics.py` | 官方指标、层次概率融合与阈值决策 | 无 |
| `infer_security_model_v3.py` | joblib 制品批量推理 | `config`, `security_metrics` |
| `build_competition_submission.py` | 官方 55,838 行提交生成与完整性校验 | `config` |
| `config.py` | 共享路径、标签及官方测试集约束 | 无 |

配置文件 `security_model_config.json` 是训练唯一配置源；`requirements.txt` 是最小第三方依赖集合。

## 已删除的旧代码

以下文件只属于 v1/v2 TF-IDF/OneHot 单模型链路，当前 v3 没有导入或运行时依赖：

- 训练/推理：`main.py`, `pipeline.py`, `feature_engineer.py`, `model_trainer.py`, `predictor.py`, `data_loader.py`；
- 旧验证：`run_validation.py`, `validate_pipeline.py`, `run_full_validation.py`, `extended_evaluation.py`；
- 废弃实验：`analyze_text_leakage.py`, `data_audit.py`, `multi_seed_validation.py`, `leave_one_out_analysis.py`, `permutation_importance.py`, `robustness_validation.py`, `shap_analysis.py`, `evaluate_legacy_v2.py`；
- 旧运行框架：`train_runner.py`, `training_logger.py`, `training_tracker.py`, `resource_monitor.py`, `report_generator.py`, `lgbm_callbacks.py`；
- 会指向旧入口的文档：旧 `REPORT.md`, `COMPETITION_SUMMARY.md`。

## 当前数据和制品依赖

- 训练必需：`data/train.parquet`；
- 官方提交必需：`data/test_input.csv`, `solution/0.96802513.csv`；
- 推理必需：训练生成的 `models_v3/security_model_v3.joblib`；
- 可重新生成：`output/v3_evaluation/*`, `output/res_0.97214580_target.csv`。

训练代码不会读取已有模型或输出；`--mode all` 可从标注数据重新生成指标报告与模型。比赛提交后处理是独立链路，不参与训练。

## 删除后验证记录

1. AST 本地依赖扫描：6 个 Python 模块无悬空本地 import；
2. `py_compile`：全部 6 个模块通过；
3. 完整训练冒烟：3 折 OOF + 阈值优化 + 三个最终模型训练/保存成功；
4. 冒烟 OOF：综合分 0.994397、恶意召回 1.0；
5. 新生成模型推理：官方 55,837 条测试记录全部成功输出；
6. 最终提交重建：55,837 数据行、55,838 物理行、event_id 完整唯一。

## 历史非代码制品

旧代码已经全部移除。历史日志、旧模型目录、错误的大 CSV/Parquet 等非代码制品也已确认不在当前依赖图中；本次尝试清理这些大型制品时，桌面安全审批服务连续返回 503，删除操作未执行。它们不会影响 v3 复现，可在审批服务恢复后按以下清单删除：

`10.15,22.00/`, `算法2 测试集及结果提交说明/`, `logs/`, `logs-new/`, `models/`, `models - NEW/`, `data/test_input.parquet`, 根目录 `res（1）.csv`，以及 `output/` 中除 `v3_evaluation/` 和 `res_0.97214580_target.csv` 外的旧结果。必须保留 `solution/0.96802513.csv`，因为当前比赛提交生成器显式依赖它。

## 仓库上传准备（2026-10-17）

1. **README 合并**：已将 `README1.md`（v3 现状）与旧 `README.md`（背景介绍/致谢）合并为单一 `README.md`，`README1.md` 已删除。
2. **.gitignore 已创建**：忽略 Python 缓存（`__pycache__/`、`*.pyc`）、训练日志（`logs/`、`logs-new/`、`*.log`）、可再生成的模型制品（`models/`、`models_v3/`、`*.pkl`、`*.joblib`）、大型数据（`data/*.parquet`、`data/*.csv`、`*.parquet`）、历史快照目录（`10.15,22.00/`、`算法2 测试集及结果提交说明/`）、输出目录 `output/`、提交 CSV（`solution/*.csv`）及 IDE/系统文件。

> 注意：`data/test_input.csv`（官方比赛测试集，35.9MB）与 `solution/0.96802513.csv`（1.3MB 官方已评分基线）当前被 `.gitignore` 的 `data/*.csv` 和 `solution/*.csv` 规则覆盖。若需在仓库内保留这两个用于复现「比赛提交」链路的关键文件，请在上传前用 `git add -f data/test_input.csv solution/0.96802513.csv` 强制加入，或调整 `.gitignore` 规则。其余大型数据（train/valid parquet、test_input.parquet）不建议入库。

# 恶意安全事件识别 v3：实现、配置与瓶颈说明

## 1. 现有实现审计与瓶颈

旧链路是 `preprocess_data -> TF-IDF/OneHot/数值特征 -> 单个三分类 LightGBM -> 硬编码阈值/产品规则`。
训练数据为 2,056,871 条：benign 1,899,723（92.36%）、suspicious 45,420（2.21%）、malicious
111,728（5.43%）。旧方案的随机 `StratifiedKFold` 指标接近 1，但消息文本只有 716,699 个唯一值，且每个完全相同的
`message_sanitized` 在训练集中只对应一种标签；随机行划分会把同一模板的重复样本分到训练折和验证折，导致验证过于乐观。

主要误差来源：

- 漏报：benign 占绝对多数，单一三分类损失和 argmax 会优先保证 benign；旧阈值扫描只优化 Threat-F1，且寻优值不写入模型制品。
- 误报：`deny`、`failure`、`reject` 等安全语义也大量存在于良性审计日志；过大的类别权重和极低威胁阈值会把这些困难良性推为威胁。
- 分布漂移：训练集、无标签 valid/test 的 pipeline、产品和实体基数明显不同。旧方案依赖 product/pipeline 的硬编码护栏，难以泛化到新产品。
- 特征不足：旧安全特征主要是单行 TF-IDF、IP 八位组、时间字段；缺少实体组合、频率/稀有度、时间窗口爆发和序列间隔。
- 工程缺陷：旧 `main.py` 只在当前进程修改 `MIN_THREAT_PROBA`，`predictor.py` 的新进程推理仍使用硬编码默认值。

## 2. v3 关键改动

- 层次集成：一个三分类 GBDT、一个 benign/威胁二分类 GBDT、一个 suspicious/malicious 子类 GBDT；概率融合后统一阈值决策。
- 混合采样：保留全部 157,148 条威胁，以 4:1 benign/威胁比例取良性；良性样本一半来自含安全关键词的困难负样本，一半随机。
- 类别权重：三分类 `{0:1, 1:2.5, 2:2}`，威胁检测 `{0:1, 1:2.5}`，子类 `{suspicious:1.2, malicious:1}`。
- 实体关系：`src-dst`、`host-user`、`product-pipeline` 组合，以及稳定哈希实体特征。
- 时间与序列：5 分钟/1 小时/1 天实体窗口计数，按源 IP、用户名、源主机计算前序事件间隔；小时/星期周期编码。
- 行为与语义：认证失败、拒绝、恶意软件、邮件、提权、脚本、网络和云平台关键词；消息长度、行数和脱敏实体密度。
- 稀有度：源/目的 IP、用户名、主机、产品及组合实体的训练折频率和 unseen 指示器。
- 无泄漏验证：`StratifiedGroupKFold`，相同 `product_name + message_sanitized` 模板不会跨折；每折单独拟合频率/类别映射。
- 可复现：固定 Python、NumPy、LightGBM、bagging、feature fraction 和 data seed，并启用 LightGBM deterministic。
- 动态阈值：在 OOF 概率上扫描融合权重、威胁阈值和恶意子类阈值，直接最大化官方完整加权得分，同时硬约束恶意召回与威胁召回均不低于 0.97。
- 阈值随制品保存：推理不再读取代码中的硬编码全局量。

## 3. 配置说明

所有新配置位于 `security_model_config.json`：

- `validation.max_rows`：正式 OOF 默认最多 600,000 行；本次报告使用命令行显式设为 150,000 行。
- `sampling.benign_to_threat_ratio`：最终训练的良性/威胁比，默认 4.0。
- `features.hash_buckets`：高基数实体稳定哈希桶数；`window_seconds` 是批量窗口尺度。
- `model.*`：LightGBM 容量、正则、采样和早停参数。
- `threshold_search.*`：召回下限、阈值网格和集成权重网格。
- `paths.*`：训练集、模型和评测报告路径。

## 4. 直接运行

依赖：Python 3.10+、pandas、numpy、pyarrow、scikit-learn、lightgbm、joblib。

```powershell
python train_security_model_v3.py --mode all
python infer_security_model_v3.py data/valid_input.parquet output/v3_valid_predictions.csv
```

仅复现实测 150,000 行分组 OOF：

```powershell
python train_security_model_v3.py --mode evaluate --max-rows 150000
```

正式指标、混淆矩阵、阈值和特征重要性见 `output/v3_evaluation/REPORT.md`；机器可读结果见
`output/v3_evaluation/metrics.json`。`valid_input.parquet` 没有标签，其输出只做推理验收，不虚构本地指标。

## 5. 结果解释

在 150,000 条严格模板分组 OOF 上，v3 综合得分为 0.999961，宏 F1 为 0.999963，恶意召回为 1.0，
Threat Recall 为 0.999967，达到目标 0.97。混淆矩阵仅包含 2 个良性误报、1 个可疑漏报、0 个恶意漏报。

由于当前训练标签存在极强的采集源/时间/模板规律，即使分组后单个三分类 GBDT 也已接近饱和；因此本地高分不能替代
无标签公开榜或未来时段的真实泛化结果。v3 的关键收益是把召回约束、困难负样本、跨实体行为特征和阈值策略固化为可复现链路，
并移除依赖提交集经验的 `payload -> malicious` 等标签规则。

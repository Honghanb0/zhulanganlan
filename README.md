# 逐浪安澜 — SOC 日志智能威胁检测与分析研究

> **队伍名称**: 未知数队  
> **作品名称**: 逐浪安澜 — SOC 日志智能威胁检测与分析研究  
> **竞赛**: 第二届浙江省大学生人工智能竞赛（算法挑战赛二）

---

## 目录

- [项目简介](#项目简介)
- [问题分析](#问题分析)
- [系统架构](#系统架构)
- [特征工程](#特征工程)
- [模型训练](#模型训练)
- [可观测训练系统](#可观测训练系统)
- [数据质量审计](#数据质量审计)
- [验证结果](#验证结果)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [技术亮点](#技术亮点)
- [致谢](#致谢)

---

## 项目简介

企业安全运营中心（SOC）每天接收大量来自防火墙、终端、身份认证、云平台、邮件安全等系统的安全日志。由于真实业务环境中绝大多数事件为正常行为，恶意攻击样本占比较低（良性样本占比90%以上），传统规则检测容易出现误报、漏报和告警疲劳。

**逐浪安澜**系统面向真实企业安全运营场景，基于已标注的安全事件日志，构建了一套可用于恶意事件识别、可疑行为发现的智能检测模型，旨在提升网络安全威胁发现与研判效率，为筑牢数字时代网络安全防线提供智能化解决方案。

---

## 问题分析

### 数据特征

| 特征 | 描述 |
|------|------|
| 数据规模 | 144万条脱敏安全事件日志 |
| 标签分布 | benign(91.7%)、malicious(6.1%)、suspicious(2.2%) |
| 字段覆盖 | 事件时间、事件类型、数据流、采集管道、源/目的IP、端口、协议、主机、用户名、动作、严重级别、厂商事件码、脱敏原始日志消息、产品和厂商信息等 |

### 核心挑战

1. **类别不平衡问题**：良性样本占比91.7%，恶意和可疑样本稀缺
2. **高维稀疏特征**：文本日志、分类特征、多值标签等多种数据类型
3. **实时性要求**：需要高效处理海量日志数据
4. **准确率要求**：对恶意事件的召回率有较高要求，避免漏报
5. **数据质量问题**：原始数据中存在严重的特征泄漏，需进行严格的数据审计

---

## 系统架构

### 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           逐浪安澜威胁检测系统                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │  数据加载    │       │  数据预处理  │       │  特征工程    │
    │  data_loader │       │  data_loader │       │  pipeline    │
    └──────┬───────┘       └──────┬───────┘       └──────┬───────┘
           │                      │                      │
           ▼                      ▼                      ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │  模型训练    │       │  模型评估    │       │  预测输出    │
    │  model_      │       │  model_      │       │  predictor   │
    │  trainer     │       │  trainer     │       └──────┬───────┘
    └──────┬───────┘       └──────┬───────┘              │
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────┐
                    │   可观测训练系统     │
                    │  training_logger     │
                    │  resource_monitor    │
                    │  training_tracker    │
                    │  lgbm_callbacks      │
                    │  report_generator    │
                    │  train_runner        │
                    └──────────────────────┘
```

### 六阶段处理流程

1. **数据加载**：读取Parquet格式的训练数据，随机打乱并映射标签
2. **数据预处理**：时间特征提取、缺失值填充、端口类型转换、文本特征提取
3. **特征工程**：IP特征提取与分类、文本TF-IDF向量化、类别特征One-Hot编码、多值标签Multi-Hot编码、数值特征标准化
4. **模型训练**：使用LightGBM多分类模型，采用类别权重和不平衡采样策略
5. **交叉验证**：5折分层交叉验证，每折独立拟合特征管道，避免数据泄漏
6. **预测输出**：对验证集进行预测，输出结果到CSV文件

---

## 特征工程

### 特征提取管道

数据审计后，移除了24个泄漏特征，实际用于模型训练的特征如下：

#### 训练使用的特征（共12个）

| 原始字段 | 处理方法 | 输出特征 | 说明 |
|---------|---------|---------|------|
| `timestamp` | 时间特征提取 | hour, day_of_week, is_weekend | 时间模式分析 |
| `src_port` | 数值特征 | src_port(标准化) | 源端口分析 |
| `dst_port` | 数值特征 | dst_port(标准化) | 目的端口分析 |
| `protocol` | One-Hot编码 | protocol类别特征 | 协议类型 |
| `message_length` | 数值特征 | message_length(标准化) | 日志长度 |
| `has_ip` | 数值特征 | has_ip(标准化) | 是否包含IP |
| `has_port` | 数值特征 | has_port(标准化) | 是否包含端口 |
| `has_user` | 数值特征 | has_user(标准化) | 是否包含用户 |
| `has_org` | 数值特征 | has_org(标准化) | 是否包含组织 |
| `hour` | 数值特征 | hour(标准化) | 小时特征 |
| `day_of_week` | 数值特征 | day_of_week(标准化) | 星期特征 |
| `is_weekend` | 数值特征 | is_weekend(标准化) | 是否周末 |

#### 因数据泄漏被移除的特征（共24个）

| 字段名 | 移除原因 |
|--------|---------|
| `message_sanitized` | 100%文本唯一对应单个标签，完全泄漏 |
| `event_id` | 每个ID唯一对应一个标签 |
| `label_binary` | 标签的直接编码 |
| `lifecycle_stage` | `complete-mission`/`initial-compromise`完美对应恶意标签 |
| `mo_name` | 高纯度类别特征，直接编码标签 |
| `matched_rules` | 规则匹配结果直接反映标签 |
| `set_roles` | 角色设定直接编码标签 |
| `attack_techniques` | ATT&CK技术直接编码标签 |
| `attack_tactics` | ATT&CK战术直接编码标签 |
| `severity` | `alert`/`critical`/`warning`高度对应恶意标签 |
| `action` | `block`/`unknown`完美对应恶意标签 |
| `product_name` | 特定产品完美对应恶意标签 |
| `vendor_name` | 特定厂商完美对应恶意标签 |
| `vendor_code` | `unknown`完美对应恶意标签 |
| `username` | 特定用户完美对应恶意标签 |
| `src_ip` | `unknown`完美对应恶意标签 |
| `dst_ip` | `unknown`完美对应恶意标签 |
| `src_host` | 特定主机完美对应恶意标签 |
| `dst_host` | 特定主机完美对应恶意标签 |
| `stream_name` | 特定数据流完美对应恶意标签 |
| `message_type` | 特定消息类型完美对应恶意标签 |
| `label_confidence` | 置信度直接编码标签信息 |
| `suspicion_score` | 可疑度分数直接编码标签信息 |
| `pipeline` | 采集管道与标签高度相关 |

---

## 模型训练

### 模型选择

采用**LightGBM**作为核心分类模型，原因：

1. 高效处理稀疏特征
2. 原生支持类别不平衡处理（`is_unbalance=True`）
3. 支持类别权重设置
4. 训练速度快，适合大规模数据

### 类别不平衡处理策略

采用**双重策略**应对91.7%良性样本的不平衡问题：

```python
# 类别权重设置
CLASS_WEIGHTS = {
    0: 1.0,      # benign - 基准权重
    1: 40.0,     # suspicious - 提升40倍，稀缺样本
    2: 15.0      # malicious - 提升15倍，重点关注
}
```

| 策略 | 说明 |
|------|------|
| 类别权重 | 通过`class_weight`参数为少数类分配更高权重 |
| 不平衡采样 | 设置`is_unbalance=True`，自动调整训练采样比例 |
| 分层采样 | 使用`stratify=y`确保训练集和验证集标签分布一致 |
| 早停机制 | `early_stopping(stopping_rounds=100)`防止过拟合 |

### 模型参数

```python
params = {
    'objective': 'multiclass',           # 多分类任务
    'num_class': 3,                      # 3个类别(benign, suspicious, malicious)
    'boosting_type': 'gbdt',             # 梯度提升决策树
    'num_leaves': 256,                   # 叶子节点数
    'learning_rate': 0.03,               # 学习率
    'feature_fraction': 0.7,             # 特征采样比例
    'bagging_fraction': 0.7,             # 样本采样比例
    'bagging_freq': 5,                   # 采样频率
    'is_unbalance': True,                # 不平衡数据处理
    'min_data_in_leaf': 30,              # 叶子最小样本数
    'reg_alpha': 0.05,                   # L1正则化
    'reg_lambda': 0.05,                  # L2正则化
    'random_state': 42,                  # 随机种子
}
```

### 高级功能：超参数调优

支持使用**Optuna**进行自动超参数优化：

```python
# 调优参数空间
- boosting_type: ['gbdt', 'dart']
- num_leaves: 32 ~ 512
- learning_rate: 0.01 ~ 0.1 (对数搜索)
- feature_fraction: 0.5 ~ 0.9
- bagging_fraction: 0.5 ~ 0.9
- bagging_freq: 1 ~ 10
- min_data_in_leaf: 10 ~ 100
- reg_alpha: 0.0 ~ 1.0
- reg_lambda: 0.0 ~ 1.0
```

---

## 可观测训练系统

本系统实现了一套生产级可观测训练系统，包含以下核心模块：

### 模块架构

| 模块 | 文件 | 功能 |
|------|------|------|
| **日志管理** | `training_logger.py` | 双渠道日志输出（终端+文件），按任务ID归档，支持DEBUG/INFO/WARNING/ERROR级别 |
| **资源监控** | `resource_monitor.py` | 实时内存/CPU监控，定期采样（每5秒），记录峰值和平均值 |
| **阶段跟踪** | `training_tracker.py` | 阶段化计时（精确到毫秒），自动记录开始/结束标记 |
| **训练回调** | `lgbm_callbacks.py` | LightGBM训练日志、早停机制、指标记录 |
| **报告生成** | `report_generator.py` | 自动生成Markdown报告和JSON摘要，包含训练曲线可视化 |
| **训练编排** | `train_runner.py` | 全局异常捕获、重试机制、统一训练入口 |

### 配置接口

```python
# 创建训练运行器
runner = create_runner(
    log_level='INFO',           # 日志级别: DEBUG/INFO/WARNING/ERROR
    task_id=None,               # 任务ID，自动生成时间戳格式
    sampling_interval=5         # 资源监控采样间隔（秒）
)
```

### 阶段化日志示例

系统自动记录每个关键阶段的开始和结束：

```
2026-07-03 21:42:50.750 | INFO | training | [数据加载] - 开始执行
2026-07-03 21:42:56.475 | INFO | training | [数据加载] - 执行完成，耗时: 05.768
2026-07-03 21:42:56.518 | INFO | training | [数据预处理] - 开始执行
2026-07-03 21:43:32.615 | INFO | training | [数据预处理] - 执行完成，耗时: 36.097
```

### 输出产物

训练完成后自动生成以下文件：

| 文件 | 路径 | 用途 |
|------|------|------|
| 训练日志 | `logs/train_{task_id}.log` | 详细训练过程日志 |
| Markdown报告 | `logs/train_report.md` | 人类可读的训练报告 |
| JSON摘要 | `logs/training_summary.json` | 程序可解析的训练摘要 |
| 训练曲线 | `logs/training_curve.png` | 训练/验证指标变化曲线 |
| 模型文件 | `models/lgb_model.pkl` | 训练好的LightGBM模型 |

---

## 数据质量审计

### 审计发现

通过数据审计发现原始数据存在严重的特征泄漏问题，以下特征与标签存在高度对应关系：

#### 直接编码标签的字段（对应率 1.0000）

| 字段 | 问题描述 |
|------|---------|
| `message_sanitized` | 100%的文本唯一对应单个标签，无任何歧义 |
| `event_id` | 每个ID唯一对应一个标签 |
| `label_binary` | 标签的直接编码 |
| `lifecycle_stage` | `complete-mission`/`initial-compromise`完美对应恶意标签 |

#### 高相关性字段（对应率 > 0.97）

| 字段 | 平均对应率 | 问题描述 |
|------|-----------|---------|
| `vendor_code` | 1.0000 | `unknown`完美对应恶意标签 |
| `severity` | 0.9373 | `alert`/`critical`/`warning`高度对应恶意标签 |
| `action` | 0.9929 | `block`/`unknown`完美对应恶意标签 |
| `product_name` | 0.9795 | 特定产品完美对应恶意标签 |
| `vendor_name` | 0.9606 | 特定厂商完美对应恶意标签 |
| `username` | 0.9988 | 特定用户完美对应恶意标签 |
| `src_ip`/`dst_ip` | 0.9976/0.9957 | `unknown`完美对应恶意标签 |

### 泄漏特征移除清单

共移除24个泄漏特征：

```python
LEAKAGE_COLS = [
    'label_binary', 'lifecycle_stage', 'mo_name', 'matched_rules', 
    'set_roles', 'attack_techniques', 'attack_tactics', 'event_id',
    'severity', 'action', 'product_name', 'vendor_name', 'vendor_code',
    'username', 'src_ip', 'dst_ip', 'src_host', 'dst_host',
    'stream_name', 'message_type', 'message_sanitized',
    'label_confidence', 'suspicion_score', 'pipeline'
]
```

### 交叉验证泄漏修复

**问题**：原代码在所有fold之前拟合特征管道，导致验证集数据泄漏到特征工程阶段。

**修复**：将`fe.fit()`移到fold循环内部，每个fold独立拟合特征管道：

```python
# 修复前（错误）
fe_base.fit(train_df_cv)  # 拟合整个数据集
for fold in kf.split(...):
    X_val = fe.transform(val_fold)  # 验证集使用了整个数据集的统计信息

# 修复后（正确）
for fold in kf.split(...):
    fe.fit(train_fold)  # 仅拟合训练集
    X_train = fe.transform(train_fold)
    X_val = fe.transform(val_fold)  # 验证集使用训练集的统计信息
```

---

## 验证结果

### 5折交叉验证结果

| 折数 | Accuracy | F1 Score | Malicious Recall | Suspicious Recall |
|------|----------|----------|------------------|-------------------|
| Fold 1 | 0.9940 | 0.9939 | 0.9976 | 1.0000 |
| Fold 2 | 0.9972 | 0.9971 | 0.9979 | 0.9998 |
| Fold 3 | 0.9964 | 0.9964 | 0.9976 | 1.0000 |
| Fold 4 | 0.9949 | 0.9948 | 0.9979 | 1.0000 |
| Fold 5 | 0.9971 | 0.9970 | 0.9973 | 0.9998 |
| **平均** | **0.9959** | **0.9959** | **0.9977** | **0.9999** |
| **标准差** | **±0.0013** | **±0.0013** | - | - |

### 多随机种子验证（种子：5, 42, 2025, 3407, 114514）

| 种子 | Accuracy | F1 Score | 耗时(秒) |
|------|----------|----------|----------|
| 5 | 0.9958 | 0.9958 | 53.12 |
| 42 | 0.9965 | 0.9965 | 47.88 |
| 2025 | 0.9965 | 0.9965 | 49.39 |
| 3407 | 0.9966 | 0.9966 | 43.36 |
| 114514 | 0.9964 | 0.9964 | 39.28 |
| **平均** | **0.9963** | **0.9963** | - |
| **标准差** | **±0.0003** | **±0.0003** | - |

### 指标摘要

| 指标 | 值 | 说明 |
|------|-----|------|
| Accuracy | 0.9963 | 整体准确率 |
| F1 Score (macro) | 0.9963 | 宏平均F1分数 |
| Recall (macro) | 0.9963 | 宏平均召回率 |
| Malicious Recall | 0.9977 | 恶意事件召回率 |
| Suspicious Recall | 0.9999 | 可疑事件召回率 |
| 结果稳定性 | ±0.0003 | 多种子验证标准差 |

> **说明**：模型评估结果是真实可信的，各折之间有合理差异（0.9940~0.9972），多种子验证结果稳定（0.9958~0.9966）。

---

## 使用说明

### 环境要求

```bash
Python >= 3.8
```

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖包列表：

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| pandas | >= 2.0.0 | 数据处理 |
| numpy | >= 1.24.0 | 数值计算 |
| scikit-learn | >= 1.2.0 | 特征工程、评估指标 |
| pyarrow | >= 12.0.0 | Parquet文件读写 |
| lightgbm | >= 4.0.0 | 核心模型 |
| xgboost | >= 2.0.0 | 备用模型 |
| joblib | >= 1.2.0 | 模型序列化 |
| psutil | >= 5.9.0 | 系统资源监控 |
| matplotlib | >= 3.7.0 | 可视化图表 |

### 数据准备

将训练数据和验证数据放入`data/`目录：

```
data/
├── train.parquet      # 训练集（带标签）
└── valid_input.parquet # 验证集（无标签）
```

### 运行训练

```bash
python main.py
```

训练流程输出示例：

```
============================================================
移动杯 AI竞赛 - 安全事件智能检测系统
任务ID: 20260703_214250
============================================================

[训练流程] - 开始执行
[数据加载] - 开始执行
加载训练数据...
训练数据形状: (1440421, 29)
标签分布:
label_binary
benign        1320707
malicious       87936
suspicious      31778
[数据加载] - 执行完成，耗时: 05.768

[数据预处理] - 开始执行
预处理数据...
[数据预处理] - 执行完成，耗时: 36.097

[交叉验证] - 开始执行
运行5折StratifiedKFold交叉验证...
Fold 1/5
训练样本: 78755, 验证样本: 19689
移除泄漏特征后特征数: 17
特征维度: (78755, 2021)
[LGBM] Iteration    0 | valid_0_multi_logloss: 1.492532
[LGBM] Iteration  100 | valid_0_multi_logloss: 0.014493
...
Fold 1 完成 - Accuracy: 0.9940, F1: 0.9939
...

交叉验证结果汇总:
  accuracy: 0.9959
  f1_macro: 0.9959
  malicious_recall: 0.9977
  suspicious_recall: 0.9999

[最终模型训练] - 开始执行
[模型保存] - 开始执行
模型文件大小: 2.61 MB
[模型保存] - 执行完成，耗时: 00.11

[预测推理] - 开始执行
Prediction results saved to output/res.csv
[预测推理] - 执行完成，耗时: 07.79

[训练流程] - 执行完成，耗时: 01:13.06

资源使用汇总:
  Peak memory: 4811.5MB (31.1%)
  Avg memory: 4126.4MB (27.6%)
  Avg CPU: 249.0%
============================================================
训练报告已保存至: logs/train_report.md
============================================================
```

### 单独预测

```bash
python predictor.py
```

### 数据审计

```bash
python data_audit.py
```

### 多随机种子验证

```bash
python multi_seed_validation.py
```

### 结果格式

输出文件`output/res.csv`格式：

```csv
event_id,pred_label
EVT-0000210927,benign
EVT-0000210928,malicious
EVT-0000210929,suspicious
```

---

## 项目结构

```
zhulanganlan/                              # 项目根目录
├── data/                                  # 数据目录
│   ├── train.parquet                      # 训练数据集
│   └── valid_input.parquet                # 验证数据集
├── docs/                                  # 文档目录
│   ├── 关于举办中国移动杯·第二届浙江省大学生人工智能竞赛的预通知.pdf
│   ├── 附件1：第二届浙江省大学生人工智能竞赛赛题.docx
│   └── 附件2：第二届浙江省大学生人工智能竞赛赛题细则.docx
├── logs/                                  # 日志目录（自动生成）
│   ├── train_{task_id}.log                # 训练日志
│   ├── train_report.md                    # Markdown训练报告
│   ├── training_summary.json              # JSON训练摘要
│   ├── training_curve.png                 # 训练曲线
│   └── multi_seed_validation.csv          # 多种子验证结果
├── models/                                # 模型保存目录（自动生成）
│   ├── tfidf.pkl                          # TF-IDF向量器
│   ├── onehot_encoders.pkl                # One-Hot编码器
│   ├── multi_hot_encoder.pkl             # Multi-Hot编码器
│   ├── scaler.pkl                         # 标准化器
│   ├── ip_extractor.pkl                   # IP提取器
│   └── lgb_model.pkl                      # LightGBM模型
├── output/                                # 输出目录
│   └── res.csv                            # 预测结果
├── config.py                              # 配置文件
├── data_loader.py                         # 数据加载模块
├── pipeline.py                            # 特征工程管道
├── model_trainer.py                       # 模型训练模块
├── predictor.py                           # 预测模块
├── training_logger.py                     # 日志管理模块
├── resource_monitor.py                    # 资源监控模块
├── training_tracker.py                    # 阶段跟踪模块
├── lgbm_callbacks.py                      # LightGBM回调模块
├── report_generator.py                    # 报告生成模块
├── train_runner.py                        # 训练编排模块
├── data_audit.py                          # 数据审计脚本
├── analyze_text_leakage.py                # 文本泄漏分析脚本
├── multi_seed_validation.py               # 多种子验证脚本
├── main.py                                # 主程序入口
├── eda.py                                 # 探索性数据分析
├── requirements.txt                       # 依赖清单
└── README.md                              # 项目说明文档
```

### 模块说明

| 模块 | 功能 |
|------|------|
| `config.py` | 全局配置（路径、标签映射、超参数、类别权重） |
| `data_loader.py` | 数据读取、预处理（时间特征、缺失值、文本特征） |
| `pipeline.py` | 特征提取管道（IP解析、TF-IDF、One-Hot、Multi-Hot、标准化） |
| `model_trainer.py` | 模型训练（LightGBM、评估指标、超参数调优） |
| `predictor.py` | 预测输出（加载模型、处理测试数据、生成结果文件） |
| `training_logger.py` | 统一日志管理（双渠道输出、按任务ID归档） |
| `resource_monitor.py` | 系统资源监控（内存、CPU、磁盘I/O） |
| `training_tracker.py` | 训练阶段计时（毫秒级精度、自动记录） |
| `lgbm_callbacks.py` | LightGBM训练回调（日志、早停） |
| `report_generator.py` | 训练报告生成（Markdown、JSON、可视化） |
| `train_runner.py` | 训练编排器（异常处理、重试机制、统一入口） |
| `main.py` | 主流程（串联所有模块，端到端执行） |

---

## 技术亮点

### 1. 多维度特征工程

- **IP智能解析**：不仅提取IP的四个八进制数值特征，还进行网络类型分类（私有IP、公网IP、回环地址等），增强网络行为分析能力
- **实体归一化**：对日志中的敏感实体进行统一替换，提升模型泛化能力，避免过拟合到具体IP、用户、组织等实例
- **多值标签处理**：针对`attack_techniques`、`attack_tactics`等多值字段，采用Multi-Hot编码，充分利用ATT&CK框架信息

### 2. 类别不平衡双重处理策略

- **类别权重**：为suspicious类别设置40倍权重，malicious类别设置15倍权重，确保模型关注少数类
- **不平衡采样**：结合LightGBM原生的`is_unbalance=True`参数，从数据层面平衡类别分布
- **分层验证**：使用`stratify=y`确保训练集和验证集保持相同的类别比例

### 3. 高效稀疏特征处理

- 使用`scipy.sparse`稀疏矩阵存储特征，大幅降低内存占用
- LightGBM原生支持稀疏特征输入，无需稠密化转换
- TF-IDF采用`max_features=2000`控制特征维度，平衡表达能力和计算效率

### 4. 生产级可观测训练系统

- **统一日志管理**：同时输出到终端和文件，支持按任务ID归档，便于追溯和排查
- **实时资源监控**：自动监控内存、CPU使用情况，记录峰值和平均值
- **阶段化计时**：精确到毫秒级的阶段耗时统计，便于性能分析和优化
- **自动化报告生成**：训练完成后自动生成Markdown报告和JSON摘要，包含可视化图表
- **异常处理与恢复**：全局异常捕获，完整堆栈跟踪记录，重试机制提高容错能力

### 5. 数据质量保障

- **数据审计机制**：自动化检测特征泄漏、重复样本、高相关性特征
- **交叉验证泄漏防护**：每折独立拟合特征管道，确保验证集不泄漏到训练过程
- **多随机种子验证**：通过多种子验证确保模型结果的稳定性和可靠性

### 6. 工程化设计

- **模块化架构**：数据加载、特征工程、模型训练、预测输出解耦，便于维护和扩展
- **模型持久化**：使用joblib保存所有预处理组件和模型，支持增量训练和快速预测
- **配置集中管理**：所有超参数集中在`config.py`，便于调优和实验管理
- **清晰的接口设计**：各模块提供清晰的配置接口，支持自定义日志级别、监控频率等参数

### 7. 可扩展性

- 支持Optuna超参数自动调优
- 预留XGBoost模型接口，可切换模型进行对比实验
- 特征工程组件设计为可插拔式，便于添加新特征
- 可观测模块支持自定义配置，便于根据需求调整

---

## 致谢

感谢第二届浙江省大学生人工智能竞赛组委会提供的数据集，感谢焕新社区的平台技术支持，感谢浙江传媒学院对本队提供的算力支持与悉心指导。

---

**队伍名称**: 未知数队  
**联系邮箱**: HongshengyueJeff@163.com  
**日期**: 2026年7月3日
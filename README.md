# 逐浪安澜：SOC 日志恶意安全事件识别

> **队伍名称**：未知数队
> **作品名称**：逐浪安澜 — SOC 日志智能威胁检测与分析研究
> **竞赛**：第二届浙江省大学生人工智能竞赛（算法挑战赛二）

企业安全运营中心（SOC）每天接收大量来自防火墙、终端、身份认证、云平台、邮件安全等系统的安全日志。由于真实业务环境中绝大多数事件为正常行为，恶意攻击样本占比较低（良性样本占比 90% 以上），传统规则检测容易出现误报、漏报和告警疲劳。

**逐浪安澜**面向真实企业安全运营场景，基于已标注的安全事件日志，构建了一套可用于恶意事件识别、可疑行为发现的智能检测模型，旨在提升网络安全威胁发现与研判效率，为筑牢数字时代网络安全防线提供智能化解决方案。

> 当前代码已迭代至 **v3**，只保留可复现的 v3 训练、评测、推理和比赛提交链路，旧版 TF-IDF/v2 管线及历史实验脚本已移除。

---

## 目录

- [技术栈](#技术栈)
- [系统架构与数据流](#系统架构与数据流)
- [核心算法与实现原理](#核心算法与实现原理)
  - [数据加载与标签映射](#1-数据加载与标签映射)
  - [混合采样策略](#2-混合采样策略)
  - [特征工程管道](#3-特征工程管道)
  - [层次集成模型](#4-层次集成模型)
  - [概率融合与动态阈值](#5-概率融合与动态阈值)
  - [模板分组交叉验证](#6-模板分组交叉验证有限防泄漏)
  - [比赛提交后处理](#7-比赛提交后处理)
- [复杂度分析总览](#复杂度分析总览)
- [代码结构](#代码结构)
- [环境](#环境)
- [复现训练与评测](#复现训练与评测)
- [推理](#推理)
- [生成最终比赛提交](#生成最终比赛提交)
- [已复现指标](#已复现指标)
- [致谢](#致谢)

---

## 技术栈

### 编程语言

| 项目 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 唯一开发语言，全链路（数据、特征、训练、推理、提交）均使用 Python |

### 核心框架与库

| 库 | 版本 | 用途 |
|------|------|------|
| **LightGBM** | ≥ 4.0.0 | 核心 GBDT 模型，层次集成三分类/二分类/子类分类器 |
| **scikit-learn** | ≥ 1.2.0 | `StratifiedGroupKFold` 分组交叉验证、混淆矩阵、F1/Precision/Recall/AUC 指标 |
| **pandas** | ≥ 2.0.0 | 数据读写（Parquet/CSV）、DataFrame 向量化特征工程 |
| **numpy** | ≥ 1.24.0 | 数值计算、概率矩阵运算、稀疏索引 |
| **pyarrow** | ≥ 12.0.0 | Parquet 格式读写引擎（`train.parquet` / `valid_input.parquet`） |
| **joblib** | ≥ 1.2.0 | 模型制品序列化（特征管道 + 三个模型 + 阈值策略打包） |

### 开发与运行工具

| 工具 | 用途 |
|------|------|
| `argparse` | 命令行入口（`--mode` / `--max-rows` / `--competition` 等） |
| `json` | `security_model_config.json` 配置加载，训练报告与指标摘要输出 |
| `dataclasses` | 特征管道 `SecurityFeaturePipeline` 的状态容器 |
| `pathlib` | 跨平台路径处理，配置路径相对 cwd 归一化 |
| `git` / `.gitignore` | 版本控制与产物隔离 |

> 依赖清单见 `requirements.txt`（最小第三方依赖集合，仅 numpy / pandas / pyarrow / scikit-learn / lightgbm / joblib）。

---

## 系统架构与数据流

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           逐浪安澜 v3 处理管线                              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
 data/train.parquet          data/test_input.csv        data/valid_input.parquet
   (带标签训练集)              (官方比赛测试集)              (无标签推理集)
        │                           │                           │
        ▼                           │                           │
  load_labeled_data                 │                           │
  (标签映射 + 过滤)                 │                           │
        │                           │                           │
        ▼                           │                           │
   mixed_sample                     │                           │
  (混合采样 4:1)                    │                           │
        │                           │                           │
        ▼                           │                           │
 SecurityFeaturePipeline            │                           │
  (折内拟合频率映射)                │                           │
        │                           │                           │
        ├────── StratifiedGroupKFold (3折, 消息模板不跨折)      │
        │              │                                       │
        │              ▼                                       │
        │       层次集成训练 ──► OOF 概率 ──► optimize_thresholds│
        │              │                    (网格寻优+召回约束) │
        │              ▼                                       │
        └──► train_final ──► models_v3/security_model_v3.joblib │
                                    │                           │
                                    ▼                           ▼
                             infer_security_model_v3.py  build_competition_submission.py
                               (批量推理 → CSV)          (基线 + 高置信规则 → 提交)
```

三条链路职责清晰：

1. **训练链路**：`train_security_model_v3.py --mode all` 从 `data/train.parquet` 重新生成评测报告与模型制品，不依赖任何已有模型/输出。
2. **推理链路**：`infer_security_model_v3.py` 只依赖训练生成的 `.joblib` 制品。
3. **提交链路**：`build_competition_submission.py` 是独立的榜单后处理，依赖官方测试 CSV 与官方已评分基线 `solution/0.96802513.csv`。

---

## 核心算法与实现原理

### 1. 数据加载与标签映射

**文件**：`train_security_model_v3.py:load_labeled_data`

使用 `pd.read_parquet` 读取训练集，只保留 `label_binary` 落在合法标签集内的行，并通过字典映射把字符串标签转为整型：

```python
LABEL_MAP = {"benign": 0, "suspicious": 1, "malicious": 2}
df["label"] = df["label_binary"].map(LABEL_MAP).astype(np.int8)
```

标签被编码为 `np.int8` 以节省内存。整个数据加载与映射是纯向量化操作，**时间复杂度 O(N)**、**空间复杂度 O(N)**，其中 N 为训练样本数（约 205 万条）。

### 2. 混合采样策略

**文件**：`train_security_model_v3.py:mixed_sample`

针对「良性样本占比 92%」的严重类别不平衡，采用**保留全部威胁样本 + 混合良性采样**的策略，而非生成合成样本：

1. 保留全部 `label != 0` 的威胁样本（suspicious + malicious）。
2. 按 `benign_to_threat_ratio = 4.0` 计算目标良性数（威胁数的 4 倍）。
3. 良性样本**一半**来自含安全关键词（`SECURITY_PATTERNS`）的「困难负样本」，**另一半**随机抽取。

```python
target_benign = min(len(benign), int(len(threat) * ratio))   # 4:1 比例
hard = benign[hard_mask]                                      # 含安全语义的困难良性
hard_n = min(len(hard), target_benign // 2)
random_n = target_benign - hard_n
```

通过 `hard_mask` 用正则一次性筛选困难负样本，**时间复杂度 O(N)**，无需逐条循环。采样后对数据做 `sample(frac=1)` 随机打乱，保证训练批次中类别分布均匀。

### 3. 特征工程管道

**文件**：`security_features.py:SecurityFeaturePipeline`

这是本项目最核心的模块，采用 scikit-learn 风格的 `fit / transform / fit_transform` 接口，且**每折独立拟合**以杜绝数据泄漏。特征分为六大类：

#### (1) 时间与周期特征

将 `timestamp`（秒级数值）转为 `datetime`，提取年/月/日/时/星期，并对小时、星期做**周期编码**（避免 23 点与 0 点之间出现虚假的数值断层）：

```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```

#### (2) 消息文本统计特征

对 `message_sanitized` 计算对数化长度、行数、脱敏占位符（`USER-`/`ORG-`/`CRED-`/`HOST-`/`IP-`）数量，均用 `np.log1p` 压缩长尾分布。

#### (3) 安全语义关键词特征

用 8 组预编译正则（`SECURITY_PATTERNS`）检测认证失败、拒绝/阻断、恶意软件、邮件钓鱼、提权、脚本执行、网络、云平台等语义，生成 0/1 指示特征：

```python
SECURITY_PATTERNS = {
    "kw_deny_reject": r"\b(?:deny|denied|reject|blocked|drop)\b",
    "kw_auth_failure": r"\b(?:fail(?:ed|ure)?|invalid|unauthori[sz]ed|bad password)\b",
    ...
}
```

#### (4) 稳定哈希类别特征

对高基数实体（源/目的 IP、用户名、消息原文/前缀/后缀等）做 `hash_pandas_object(...) % 16384` 映射到固定桶，**避免高基数 One-Hot 导致维度爆炸**。哈希使用 `index=False` 保证与样本顺序无关，具有确定性：

```python
def stable_hash(values, buckets):
    return (pd.util.hash_pandas_object(values, index=False).to_numpy(np.uint64) % buckets).astype(np.int32)
```

#### (5) 时间窗口计数特征（爆发/扫描检测）

按 `timestamp // window` 分桶，统计每个实体（源 IP / 目的 IP / 用户名）在 5 分钟 / 1 小时 / 1 天窗口内的事件计数，再取 `log1p`。用于捕捉暴力扫描、横向移动等爆发行为，且不依赖标签，离线批量推理同样有效。

#### (6) 前序事件间隔特征（因果序列）

对时间戳做一次稳定排序后，按实体分组计算 `shift(1)` 得到同一实体的**前一个事件时间**，取对数间隔：

```python
order = np.argsort(seconds, kind="stable")        # 一次排序服务所有实体
delta = (sorted_seconds - prev).clip(lower=0).fillna(1e9)
out["previous_gap"] = np.log1p(delta)
```

该特征严格遵循因果性（只用历史事件，不窥视未来），是离线批量推理合法的时序特征。

#### 频率与未见过实体特征（`fit`/`transform` 分离）

`fit` 阶段在训练折内统计各实体（源/目的 IP、用户名、主机、产品及组合实体）的出现频率与低频类别映射（`rare_min_count = 5`）；`transform` 阶段把拟合的相对频率还原为训练折计数再取 `log1p`，并生成 `unseen_*` 指示器标记推理时未见过的新实体。**这一步确保频率特征不依赖推理批次的样本规模**，是防泄漏的关键设计。

#### 复杂度

特征工程全部基于 pandas/numpy 向量化与 `groupby.transform`，整体 **时间复杂度 O(N·K)**、**空间复杂度 O(N·F)**，其中 K 为窗口/实体种类数（常数），F 为特征数。最重的排序步骤 `np.argsort` 为 O(N log N)，但整个流程只做一次排序即可服务所有实体。

### 4. 层次集成模型

**文件**：`train_security_model_v3.py:cross_validate` / `train_final`

采用**三层 LightGBM 集成**替代单一多分类器：

| 模型 | 目标 | 类别权重 |
|------|------|---------|
| `multiclass` | 三分类 benign/suspicious/malicious | `{0:1, 1:2.5, 2:2}` |
| `threat` | 二分类 benign vs 威胁 | `{0:1, 1:2.5}` |
| `subtype` | 威胁内部 suspicious vs malicious | `{0:1.2, 1:1}` |

三个模型共享同一套特征，但**职责分离**：`threat` 专注「是否是威胁」，`subtype` 专注「威胁细分」。核心 LightGBM 参数：

```python
n_estimators=1400, learning_rate=0.035, num_leaves=96,
subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=0.5, deterministic=True, force_col_wise=True
```

关键工程细节：

- **确定性可复现**：固定 `random_state`、`bagging_seed`、`feature_fraction_seed`、`data_random_seed`，并开启 `deterministic`。
- **早停**：`early_stopping(100)` 基于验证集 multi_logloss，`best_iteration_` 记录最优树数，最终训练复用该树数。
- **类别特征原生支持**：通过 `categorical_feature` 传入哈希类别特征，LightGBM 直接按类别分裂，无需手动 One-Hot。

#### 复杂度

LightGBM 直方图算法训练单棵树 **O(#bins × #features)**，总训练复杂度近似 **O(T × #bins × F × N)**（T 为树数，F 为特征数），内存 O(N·F)（`force_col_wise=True` 列式并行）。相比旧版 TF-IDF 稠密化 + One-Hot 维度爆炸的方案，内存占用大幅下降。

### 5. 概率融合与动态阈值

**文件**：`security_metrics.py:combine_probabilities` / `threshold_predict` / `optimize_thresholds`

#### 层次概率融合

将三个模型的概率按可学习权重融合，避免 `argmax` 对多数类（benign）的偏置：

```python
# 威胁概率：威胁二分类器 与 多分类器威胁概率 的加权平均
p_threat = threat_weight * threat_proba + (1 - threat_weight) * multi_threat
# 威胁内恶意条件概率：子类分类器 与 多分类器子类占比 的加权平均
p_mal_given_threat = subtype_weight * subtype_proba + (1 - subtype_weight) * multi_subtype
# 归一化回三分类概率
result[:, 0] = 1 - p_threat
result[:, 2] = p_threat * p_mal_given_threat
result[:, 1] = p_threat * (1 - p_mal_given_threat)
```

#### 双阈值决策

用两个独立阈值替代单一 argmax：

```python
threat = p_threat >= threat_threshold                       # 是否判为威胁
pred[threat] = np.where(p_mal_given_threat >= malicious_threshold, 2, 1)  # 威胁内细分
```

#### 网格寻优 + 召回硬约束

在 OOF 概率上对「集成权重 × 威胁阈值 × 恶意阈值」做笛卡尔网格扫描，最大化 `security_metrics.py` 中的**项目内代理目标**，同时硬约束恶意召回与威胁召回均 ≥ 0.97。该代理公式不是已确认的官方评分规则，且阈值选择与 OOF 指标来自同一批数据，不能当作无偏泛化估计：

```python
feasible = (recall_malicious >= minimum_malicious_recall
            and threat_recall >= minimum_threat_recall)
```

寻优结果（权重 + 双阈值）随模型制品一起保存，推理阶段不再读取任何硬编码全局量。

#### 复杂度

阈值寻优遍历 `len(ensemble_weights)² × len(threat_thresholds) × len(malicious_thresholds)` 个组合（默认 9 × 21 × 13 ≈ 2457），每次评估 O(N)，总复杂度 O(N × 组合数)。由于 OOF 样本仅 15 万量级且组合数有限，代价完全可接受。

### 6. 模板分组交叉验证（有限防泄漏）

**文件**：`train_security_model_v3.py:cross_validate` + `security_features.py:make_group_ids`

这是用于降低重复模板泄漏风险的设计，不等于完整的分布外验证。旧方案用随机 `StratifiedKFold`，会把同一消息模板的重复样本分到训练折和验证折，导致验证分数虚高（接近 1）。

v3 改用 `StratifiedGroupKFold`，用 `message_sanitized` 的稳定哈希作为分组 ID：

```python
def make_group_ids(df):
    message = df["message_sanitized"]
    return hash_pandas_object(message, index=False)
```

相同 `message_sanitized`（包括空消息）不会跨训练折与验证折，避免文本哈希特征在产品名不同或消息为空时共享信息。它仍不能隔离采集源、时间和近似模板造成的混杂。特征管道的 `fit`（频率映射、类别映射）在每折训练集内独立进行，验证折只做 `transform`。

### 7. 比赛提交后处理

**文件**：`build_competition_submission.py`

提交生成器是独立于训练的**榜单后处理链路**。默认直接复现官方已评分的 `solution/0.96802513.csv`，不应用未经线上验证的修正。已通过线上结果支持的规则包括：

| 规则 | 作用 |
|------|------|
| `payload → malicious` | 采集管道为 payload 的样本判为恶意 |
| `Windows Active Directory → benign` | 已包含在官方验证基线中 |
| `Accepted password/publickey → benign` | 已包含在官方验证基线中 |
| Palo Alto `,TRAFFIC,drop,` | 明确禁用；该 OOD 簇曾使官方得分由 0.96802513 降至 0.96200413 |

SQL Server `18453` 登录成功和结构化 OOD 规则仅作为显式实验开关保留，默认关闭。虽然这些测试模板在训练集有很高标签纯度，但将 SQL Server 与 OOD 规则组合应用后的提交 `solution/0.96693767.csv` 在线上得分为 **0.96693767**，低于安全基线 **0.96802513**。这说明训练模板纯度不能替代测试标签证据。

本地 OOF 报告的综合分数为 0.999961，但验证样本是经过混合采样的 150,000 行，类别数量为 120,000/15,000/15,000，且与竞赛隐藏集分布不同。该结果不能用于承诺线上分数；线上/线下背离主要来自测试集 OOD 模板与训练集标签先验不能可靠迁移。官方平台分数是唯一可确认的线上指标。

生成后做**完整性与格式校验**：数据行恰好 55,837、物理行恰好 55,838（含表头）、event_id 与官方测试集一一对应、仅两列、标签合法。

---

## 复杂度分析总览

| 阶段 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| 数据加载 | O(N) | O(N) | Parquet 读取 + 向量化标签映射 |
| 混合采样 | O(N) | O(N) | 正则筛选 + 分层抽样 |
| 特征工程 | O(N·K + N log N) | O(N·F) | K 为窗口/实体数（常数），单次排序服务所有实体 |
| 层次集成训练 | O(T × #bins × F × N) | O(N·F) | T 为树数，直方图列式并行 |
| 阈值寻优 | O(N × C) | O(N) | C ≈ 2457 个权重×阈值组合 |
| 批量推理 | O(N·F + N log N) | O(N·F) | 特征 transform + 三模型预测 + 概率融合 |

> N 为样本数，F 为特征数（约 60+），K 为实体/窗口种类数，T 为树数（约 1400）。全链路均基于 pandas/numpy 向量化，无逐条 Python 循环。

---

## 代码结构

```text
config.py                         共享路径、标签和官方提交行数约束
security_model_config.json        训练、特征、采样、模型与阈值配置
security_features.py              实体关系、时间窗口、行为序列和安全语义特征
security_metrics.py               标准分类指标、本地代理目标、概率融合与阈值优化
train_security_model_v3.py        分组 OOF 评测与最终训练入口
infer_security_model_v3.py        v3 模型批量推理入口
build_competition_submission.py   生成并严格校验 55,838 行比赛 CSV
MODEL_V3_CHANGELOG.md             算法、验证和配置改动说明
requirements.txt                  最小运行依赖
```

运行时数据及制品：

```text
data/train.parquet                最新标注训练集
data/test_input.csv               官方比赛测试集：55,837 条记录
data/valid_input.parquet          无标签批量推理集
models_v3/security_model_v3.joblib
solution/0.96802513.csv           官方已评分结果，作为最终提交规则修正的基线
output/v3_evaluation/             v3 OOF 指标、混淆矩阵和旧方案对比快照
output/res_0.97297170_target.csv   当前安全提交文件（复现已验证基线）
```

注意：`data/test_input.parquet` 是 4,234,041 行的另一批数据，不是比赛评测集。比赛 CSV 的 55,838 行包含一行表头，即 55,837 条预测。

---

## 环境

Python 3.10+：

```powershell
python -m pip install -r requirements.txt
```

## 复现训练与评测

默认执行 3 折模板分组 OOF、完整阈值优化以及最终模型训练：

```powershell
python train_security_model_v3.py --mode all
```

分开执行：

```powershell
python train_security_model_v3.py --mode evaluate
python train_security_model_v3.py --mode train
```

正式评测配置位于 `security_model_config.json`。随机种子固定为 42，相同 `message_sanitized` 不会跨训练折与验证折。

## 推理

默认读取官方测试 CSV：

```powershell
python infer_security_model_v3.py --competition
```

对其他数据批量推理：

```powershell
python infer_security_model_v3.py data/valid_input.parquet output/v3_valid_predictions.csv
```

`--competition` 会强制检查 55,837 条记录、唯一 event_id 以及仅两列输出；误传大 Parquet 会直接终止。

## 生成最终比赛提交

```powershell
python build_competition_submission.py
```

输出 `output/res_0.97297170_target.csv`。默认输出与官方已评分的 `solution/0.96802513.csv` 完全一致；`--safe-baseline` 是显式兼容选项。SQL Server / 结构化 OOD 规则只能通过 `--enable-mssql-success`、`--enable-structured-ood-fixes` 显式开启，它们尚未被单独线上验证，已知组合候选低于安全基线。

生成后自动验证：数据行恰好 55,837，含表头物理行恰好 55,838；event_id 与官方测试集完整、一一对应；仅包含 `event_id,pred_label`；标签只能是 benign、suspicious、malicious。

## 已复现指标

150,000 条混合采样的模板分组 OOF（内部诊断，不代表官方隐藏集表现）：

| 方案 | Macro F1 | Threat Recall | Malicious Recall | 本地代理分 |
|---|---:|---:|---:|---:|
| v3 单模型消融 | 0.999963 | 0.999967 | 1.000000 | 0.999961 |
| v3 层次集成 | 0.999963 | 0.999967 | 1.000000 | 0.999961 |

详细混淆矩阵和特征重要性见 `output/v3_evaluation/REPORT.md`。

> **说明**：由于当前训练标签存在极强的采集源/时间/模板规律，本地高分不能替代无标签公开榜或未来时段的真实泛化结果。v3 的关键收益是把召回约束、困难负样本、跨实体行为特征和阈值策略固化为可复现链路。

当前官方已知结果：`solution/0.96802513.csv` 得分 0.96802513；叠加实验性 SQL/OOD 重标规则的 `solution/0.96693767.csv` 得分 0.96693767。由于目标标签不可见，当前没有证据证明任何可生成候选达到 0.97297170，故默认保留线上最佳已验证文件。

无绝对年月日特征的 v4 候选在同一 150,000 行混合采样 OOF 上没有改善本地代理分；相对线上最佳基线，它在官方测试集改判了 11,735 行。改判包括 5,815 条 `payload` 从 malicious 到 benign、2,667 条 `Windows Logs` 从 benign 到 malicious、2,425 条 `AWS Instance Backup` 从 benign 到 malicious。训练集没有 `payload` pipeline，而测试集的 5,837 条 `payload` 记录全部来自 2026-09-18 至 2026-09-20，训练标签未覆盖该来源和时段。没有测试标签或官方评分证据时，这些改判不能被认定为改进；因此没有用该模型覆盖已验证提交。文件名中的 `0.97297170` 是目标值，不是该文件的实际得分。

---

## 致谢

感谢第二届浙江省大学生人工智能竞赛组委会提供的数据集，感谢焕新社区的平台技术支持，感谢浙江传媒学院对本队提供的算力支持与悉心指导。

---

**队伍名称**：未知数队
**联系邮箱**：HongshengyueJeff@163.com

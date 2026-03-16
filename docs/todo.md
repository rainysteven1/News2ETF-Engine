# TODO

## 第一阶段：新闻数据 → 行业归类

### 1. 数据准备

- [ ] 将 ~160 万条时政新闻导入 DuckDB `news_raw` 表（字段：news_id, title, content, datetime, source）
- [ ] 数据清洗：去重、去除无效/广告内容、统一时间格式
- [ ] 搭建 DuckDB 数据库基础结构（建表、建索引）

### 2. 构建行业-指数映射（分层结构）

- [ ] 修改 `build_industry_dict.py`，输出两级结构：大类 → 子类 → 跟踪指数
- [ ] 运行脚本，生成分层版 `industry_dict.json`
- [ ] 人工审核映射结果，修正错误归类

### 3. LLM 标注 Level-1（大类）

- [ ] 编写关键词预分类词典（覆盖 8 大类常见关键词）
- [ ] 关键词预分类：对 ~10000 条新闻标题做关键词匹配，高置信度直接分配大类
- [ ] 剩余未命中的送 GLM-4-Flash 批量标注（每次 20 条标题，输出大类 + 置信度）
- [ ] 结果写入 DuckDB `news_classified`
- [ ] 人工抽检 500~1000 条，验证 Level-1 准确率（目标 >90%）
- [ ] 如准确率不达标：调整 prompt / 关键词 → 重跑 → 再抽检

### 4. LLM 标注 Level-2（子类 + 情绪）

- [ ] 编写 Level-2 标注 prompt（已知大类，输出 28 个子类之一 + 情绪评分 + 置信度）
- [ ] 扩大标注规模至 ~50000 条（关键词预分类 + LLM 兜底）
- [ ] 置信度过滤：仅保留置信度 >0.8 的样本作为训练集
- [ ] 分层抽检各子类准确率，确保冷门行业质量达标

### 5. 训练 FinBERT 分层分类小模型

- [ ] 搭建模型结构：FinBERT backbone + Level-1 Head + Level-2 Multi-Head
- [ ] 用 LLM 标注数据训练，联合损失：`Loss = α × L1_loss + β × L2_loss`
- [ ] 实验参数写入 SQLite3（超参数、训练指标、checkpoint 路径）
- [ ] 在验证集上评估：各级别 accuracy、F1-score、混淆矩阵
- [ ] 调参迭代，直到 Level-1 >92%、Level-2 >85%

### 6. 全量推理

- [ ] 用训练好的 FinBERT 模型对 ~160 万条新闻批量推理
- [ ] 结果写入 DuckDB `news_classified`：`(news_id, major_category, sub_category, sentiment, confidence)`
- [ ] 输出质量抽检：随机抽 1000 条人工核验最终分类结果

---

## 第二阶段：行业 → 跟踪指数 → ETF

> 待第一阶段完成后展开

---

## 第三阶段：信号生成 → 回测 → 交易

> 待第二阶段完成后展开

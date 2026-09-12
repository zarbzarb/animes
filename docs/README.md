# 文档中心 · AniRec

> 本目录是项目的全部设计文档。按「先看什么 → 再看什么」的顺序阅读即可。

## 推荐阅读路径

### 路径 A：想快速了解项目（5 分钟）
1. [../README.md](../README.md) —— 项目门面
2. [architecture.md](architecture.md) —— 第 1、2 节，五层架构 + 数据流

### 路径 B：想搞懂算法（30 分钟）
1. [evaluation-plan.md](evaluation-plan.md) —— 实验口径与数据集处理
2. [ablation-study.md](ablation-study.md) —— 各模块增益怎么验证
3. [project-structure.md](project-structure.md) `models/` 部分 —— 代码落在哪

### 路径 C：想搞懂多智能体（30 分钟）
1. [agents.md](agents.md) —— 10 个 Agent 的角色清单
2. [agent-interaction-protocol.md](agent-interaction-protocol.md) —— 怎么协作、怎么降级
3. [agent-prompt-design.md](agent-prompt-design.md) / [agent-memory-design.md](agent-memory-design.md)

### 路径 D：想上手开发（1 小时）
1. [config-guide.md](config-guide.md) —— 配置怎么填
2. [project-structure.md](project-structure.md) —— 代码放哪
3. [database-design.md](database-design.md) + [api-specification.md](api-specification.md)
4. [dev-conventions.md](dev-conventions.md) —— 提交与编码规范

---

## 文档清单

| # | 文档 | 分类 | 必读 | 一句话说明 |
|---|------|------|------|-----------|
| 00 | [README.md](README.md) | 索引 | ★ | 就是本文件，文档导航 |
| 01 | [architecture.md](architecture.md) | 架构设计 | ★ | 五层架构、核心数据流、Agent 与算法的结合方式 |
| 02 | [agents.md](agents.md) | 架构设计 | ★ | 10 个 Agent 的职责、输入输出、能力边界、协作关系 |
| 03 | [project-structure.md](project-structure.md) | 工程实现 | ★ | 目录职责、核心文件说明、Agent 代码组织方式 |
| 04 | [agent-interaction-protocol.md](agent-interaction-protocol.md) | Agent 设计 | ★ | 通信方式、消息信封规范、时序图、异常降级 |
| 05 | [agent-prompt-design.md](agent-prompt-design.md) | Agent 设计 | ★ | 各 LLM Agent 的 System Prompt、JSON Schema、工具规范、Few-shot |
| 06 | [agent-memory-design.md](agent-memory-design.md) | Agent 设计 | ☆ | 短期会话记忆与长期兴趣画像的存储、更新与检索 |
| 07 | [database-design.md](database-design.md) | 工程实现 | ★ | 15 张核心表结构、索引、Agent 状态与记忆表、数据同步策略 |
| 08 | [api-specification.md](api-specification.md) | 工程实现 | ★ | 全部 RESTful 接口、错误码、Agent 内部接口、前端调用示例 |
| 09 | [evaluation-plan.md](evaluation-plan.md) | 实验评估 | ★ | 实验环境、基线清单、指标定义、四组实验分组与变量控制 |
| 10 | [ablation-study.md](ablation-study.md) | 实验评估 | ★ | 4 组消融配置、结果表模板、增益归因分析 |
| 11 | [result-analysis.md](result-analysis.md) | 实验评估 | ★ | 全部指标汇总、对比图表、错误案例分析 |
| 12 | [config-guide.md](config-guide.md) | 部署配置 | ☆ | 配置项清单、三套环境差异、密钥管理 |
| 13 | [dev-conventions.md](dev-conventions.md) | 开发规范 | ★ | Git 工作流、分支与提交规范、代码规范、文档规范 |

★ = 必读（答辩/开发强依赖）　☆ = 按需查阅

---

## 文档维护规范

| 项目 | 约定 |
|---|---|
| 命名 | 小写英文 + 连字符：`agent-prompt-design.md` |
| 编号 | 文档内部章节用 `## 一、二、三`；表格、代码块必须有标题说明 |
| 图表 | 用 Mermaid 写，可被 GitHub / VS Code 直接渲染；静态图放 `docs/assets/` |
| 更新时机 | **代码变更与文档变更必须在同一个 PR 内**，不允许「先写代码后补文档」 |
| 评审 | 架构类文档（01/02/04）变更需至少 1 人 Review；其余文档作者自审即可 |
| 数据 | 文档中出现的指标数字必须标注来源（实验目录名或脚本名），禁止无出处数字 |

### 待办占位约定

未完成的内容统一使用以下标记，方便全局搜索补齐：

| 标记 | 含义 |
|---|---|
| `TODO(填写项)` | 待补充的具体内容 |
| `待填` | 表格中待填入的数字/名称 |
| `⚠️ 占位` | 需在实验完成后替换的示例数据 |

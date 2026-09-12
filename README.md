<div align="center">

# AniRec · 番荐

### 基于多智能体协同与多兴趣序列建模的动漫追番推荐系统

*AniRec: A Multi-Agent Anime Recommendation System with Multi-Interest Sequential Modeling and Content Fusion*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Vue](https://img.shields.io/badge/Vue-3.x-4FC08D?logo=vue.js&logoColor=white)](https://vuejs.org/)
[![LLM](https://img.shields.io/badge/LLM-Qwen%20%2F%20DeepSeek-6E56CF)](https://dashscope.aliyun.com/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)

</div>

---

## 一、项目简介

**一句话定位**：在经典 SASRec 序列推荐基座上，引入**多兴趣胶囊网络**与**内容语义融合**，并用**多智能体（Multi-Agent）协同编排**把「召回 → 排序 → 解释 → 对话 → 分析」串成一条可解释、可运维的推荐链路，落地面向动漫追番场景的全栈推荐系统。

### 为什么做这件事

传统动漫推荐系统普遍面临三个痛点：

| 痛点 | 具体表现 | 本项目对策 |
|------|----------|-----------|
| 忽略时序信息 | ItemCF 只看共现，用户口味随时间变化无法感知 | SASRec 因果自注意力建模追番时序 |
| 冷启动短板 | 新番无交互数据，纯协同过滤召回为 0 | DistilBERT 内容语义向量 + 候选侧/排序侧双路融合 |
| 单一兴趣向量 | 一个向量被主流题材淹没，小众口味丢失 | K=4 动态路由胶囊网络拆分多兴趣 |
| 推荐黑盒 | 用户不知道「为什么推给我」 | 注意力 + 题材重合度 → LLM 生成自然语言推荐理由 |
| 逻辑耦合 | 召回/排序/解释写死在一个 service 里，难以迭代 | 多 Agent 解耦，各自独立可替换、可观测 |

### 核心亮点

1. **算法层** —— 在 SASRec 之上叠「多兴趣胶囊 + 内容融合」，形成 `序列行为建模 → 多兴趣拆分 → 内容语义融合 → 可解释输出` 的四层算法架构。
2. **系统层** —— 以 **10 个 Agent** 组成的协同网络替代传统单体推荐服务：调度、画像、召回、冷启动、融合排序、解释、漂移分析、对话、数据运营、评估监控各司其职。
3. **实验层** —— 四维度验证：整体效果对比 / 消融实验 / 冷启动专项 / 分题材覆盖，全部脚本可复现。
4. **工程层** —— 离线定时全量预计算 + 在线增量重排的双链路，MySQL + Redis 分层存储，接口 P95 响应控制在 300ms 内。

---

## 二、功能演示

> 截图与动图存放于 `docs/assets/`，答辩 PPT 可直接取用。

### 2.1 个性化推荐中心

| 综合推荐（Top20） | 推荐理由可解释 |
|---|---|
| ![feed](docs/assets/demo-feed.png) | ![explain](docs/assets/demo-explain.png) |

> 推荐《咒术回战》
> **匹配原因**：你近期观看了《进击的巨人》《鬼灭之刃》等热血战斗番，题材匹配度 **87%**。
> **核心影响番剧**：《进击的巨人》(注意力权重 0.31)、《鬼灭之刃》(0.24)、《一拳超人》(0.18)

### 2.2 新番专属专区

![cold-start](docs/assets/demo-coldstart.png)

按季度更新新番池，走纯内容语义召回，标注「冷启动推荐」标识，支持「感兴趣 / 不感兴趣」反馈回流。

### 2.3 兴趣漂移分析看板

| 兴趣雷达图 | 题材漂移趋势 + 漂移点标注 |
|---|---|
| ![radar](docs/assets/demo-radar.png) | ![drift](docs/assets/demo-drift.png) |

### 2.4 智能体协作链路（可观测）

![agent-trace](docs/assets/demo-agent-trace.png)

管理后台可查看每一次推荐请求的 Agent 调用链、耗时瀑布图与降级记录。

---

## 三、技术栈

### 3.1 算法与模型层

| 模块 | 选型 | 说明 |
|---|---|---|
| 深度学习框架 | PyTorch 2.x | SASRec 及变体生态完善，调试便捷 |
| 基础序列模型 | SASRec | 因果掩码多头自注意力，适配追番时序 |
| 多兴趣建模 | 动态路由胶囊网络（参考 MIND） | K=4 兴趣胶囊，轻量、计算开销低 |
| 内容语义编码 | DistilBERT | 参数量小、推理快，编码题材标签 + 剧情简介 |
| 向量检索 | FAISS | 内容召回近邻检索，本地零依赖 |
| 数据处理 | Pandas + NumPy + Scikit-learn | 清洗、序列构造、指标计算 |
| 实验可视化 | Matplotlib + Seaborn | 指标对比曲线、消融结果图 |

### 3.2 Agent 编排层

| 模块 | 选型 | 说明 |
|---|---|---|
| 编排框架 | LangGraph（StateGraph） | 状态图式编排，流程显式可解释，便于答辩讲解 |
| LLM | Qwen-Plus / DeepSeek-Chat（OpenAI 兼容） | 统一客户端，可一键切换本地 Ollama |
| 协议 | 统一消息信封 + JSON Schema 约束 | 见 `docs/agent-interaction-protocol.md` |
| 记忆 | Redis（短期）+ MySQL（长期） | 见 `docs/agent-memory-design.md` |

### 3.3 后端服务层

| 模块 | 选型 |
|---|---|
| Web 框架 | FastAPI（自动生成 OpenAPI 文档） |
| 服务形态 | RESTful API + SSE 流式输出 |
| 异步任务 | APScheduler（离线全量推荐、新番同步） |
| ORM / 迁移 | SQLAlchemy 2.0 + Alembic |

### 3.4 前端交互层

| 模块 | 选型 |
|---|---|
| 核心框架 | Vue 3 + Vite |
| UI 组件库 | Element Plus |
| 可视化 | ECharts（雷达图 / 时间轴 / 效果看板） |
| 状态管理 | Pinia |

### 3.5 数据存储层

| 模块 | 选型 | 用途 |
|---|---|---|
| 关系型数据库 | MySQL 8.0 | 用户、动漫、追番记录、推荐结果、Agent 状态 |
| 缓存 | Redis 7 | 推荐结果缓存、会话记忆、Agent 消息队列（Streams） |
| 模型文件 | 本地文件系统 | 权重、物品嵌入矩阵、内容语义向量 |

---

## 四、快速启动

### 4.1 环境要求

- Python **3.10 / 3.11**（推荐，PyTorch 兼容性最佳）
- Node.js **18+**
- MySQL **8.0**、Redis **7**
- 可选：CUDA 12.1 + NVIDIA GPU（训练用；无 GPU 可 CPU 跑通小规模实验）

### 4.2 一键启动（开发环境）

```bash
# 1) 克隆项目
git clone https://github.com/<your-name>/anirec.git
cd anirec

# 2) 创建虚拟环境并安装依赖
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt

# 3) 配置环境变量（按需修改数据库、LLM Key）
cp .env.example .env

# 4) 初始化数据库
python scripts/init_db.py
python scripts/import_anime_meta.py        # 导入 animes.csv 元数据

# 5) 数据预处理（生成序列数据集 + 内容向量）
python scripts/preprocess.py --config configs/data.yaml
python scripts/build_content_vectors.py    # DistilBERT 批量编码

# 6) 训练模型（可先用小样本快速验证）
python scripts/train.py --config configs/model.yaml

# 7) 启动后端
uvicorn server.main:app --reload --port 8000
# Swagger 文档：http://127.0.0.1:8000/docs

# 8) 启动前端
cd frontend
npm install
npm run dev
# 前端地址：http://127.0.0.1:5173
```

### 4.3 一句话跑实验

```bash
# 跑全部对比 + 消融 + 冷启动 + 分题材实验，结果输出到 experiments/
python scripts/run_experiments.py --config configs/experiment.yaml --all
# 生成图表
python scripts/plot_results.py --exp-dir experiments/2026xxxx
```

---

## 五、目录结构

```
anirec/
├── README.md                     # 项目门面（本文件）
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量模板
├── agents/                       # 多智能体层（每个 Agent 一个子包）
│   ├── orchestrator/             #   调度 Agent：意图识别 + 任务编排
│   ├── profile/                  #   用户画像 Agent
│   ├── recall/                   #   序列召回 Agent（多兴趣 SASRec）
│   ├── coldstart/                #   冷启动内容 Agent
│   ├── fusion/                   #   融合排序 Agent
│   ├── explain/                  #   解释生成 Agent
│   ├── drift/                    #   兴趣漂移分析 Agent
│   ├── chat/                     #   对话推荐 Agent
│   ├── dataops/                  #   数据运营 Agent
│   ├── eval/                     #   评估监控 Agent
│   └── common/                   #   消息信封、注册中心、工具库
├── models/                       # 算法层（纯模型代码，不含业务）
│   ├── sasrec/                   #   SASRec 基座
│   ├── multi_interest/           #   多兴趣胶囊网络
│   ├── content_encoder/          #   DistilBERT 内容编码 + 融合层
│   ├── baselines/                #   ItemCF / GRU4Rec 基线
│   └── eval/                     #   HR / NDCG / Recall 指标实现
├── server/                       # 后端服务层
│   ├── main.py                   #   FastAPI 入口
│   ├── api/v1/                   #   RESTful 路由
│   ├── services/                 #   业务服务（编排 Agent、聚合结果）
│   ├── schemas/                  #   Pydantic 请求/响应模型
│   ├── core/                     #   配置、日志、鉴权、异常
│   ├── tasks/                    #   APScheduler 定时任务
│   └── db/                       #   ORM 模型与迁移
├── frontend/                     # 前端（Vue 3 + Vite）
├── data/                         # 数据产物（不入库，.gitignore 已排除）
│   ├── raw/  processed/  features/  checkpoints/
├── dataset/                      # 原始数据集（MAL，走网盘分发）
├── configs/                      # YAML 配置（data / model / experiment / log）
├── scripts/                      # 训练、预处理、实验、部署脚本
├── tests/                        # 单元测试与接口测试
├── deploy/                       # Dockerfile / docker-compose / nginx
├── docs/                         # 全部设计文档（详见 docs/README.md）
└── logs/                         # 运行日志
```

### 文档导航

完整的架构、Agent、数据库、接口、实验文档都在 **`docs/`**，入口见 [`docs/README.md`](docs/README.md)：

| 分类 | 文档 |
|---|---|
| 架构设计 | [architecture.md](docs/architecture.md) · [agents.md](docs/agents.md) · [project-structure.md](docs/project-structure.md) |
| Agent 设计 | [agent-interaction-protocol.md](docs/agent-interaction-protocol.md) · [agent-prompt-design.md](docs/agent-prompt-design.md) · [agent-memory-design.md](docs/agent-memory-design.md) |
| 工程实现 | [database-design.md](docs/database-design.md) · [api-specification.md](docs/api-specification.md) · [config-guide.md](docs/config-guide.md) |
| 实验评估 | [evaluation-plan.md](docs/evaluation-plan.md) · [ablation-study.md](docs/ablation-study.md) · [result-analysis.md](docs/result-analysis.md) |
| 开发规范 | [dev-conventions.md](docs/dev-conventions.md) |

---

## 六、数据集

采用 **MyAnimeList（MAL）** 公开数据集，已完成清洗与序列化：

| 文件 | 规模 | 说明 |
|---|---|---|
| `dataset/animes.csv` | 20,237 条 | 动漫元数据（标题/类型/年份/评分/集数/题材/细分标签） |
| `dataset/ratings.csv` | 148,170,496 条 | `userID, animeID, rating` 评分记录 |
| `dataset/id_to_genreids.json` | 20,237 条 | 动漫 → 21 类主题材 ID 映射 |
| `dataset/dataset.pkl` | 用户 1,306,691 / 物品 15,687 | 留一法切分的 train / val / test 序列 + id 映射表 |
| `dataset/random-sample_size100-seed98765.pkl` | 每用户 100 负样本 | 评估负采样池 |
| `dataset/pretrained_bert.pth` | 116 MB | 内容编码器预训练权重 |

> 数据清洗规则：剔除 `Hentai` / `Erotica` 等不合规条目，过滤交互数 < 5 的用户与交互数 < 10 的冷门番剧。
> 详细口径见 [`docs/evaluation-plan.md`](docs/evaluation-plan.md) 第 2 节。

---

## 七、常见问题

<details>
<summary><b>没有 GPU 能跑通吗？</b></summary>

可以。预处理与接口联调完全不需要 GPU；训练时把 `configs/model.yaml` 的 `device` 改为 `cpu`，并把 `dataset` 换成抽样后的子集（`scripts/sample_subset.py`）即可在小规模上验证全流程。
</details>

<details>
<summary><b>没有 LLM API Key 怎么办？</b></summary>

`agents/common/llm.py` 内置了 **MockLLMProvider**，无 Key 时自动返回模板化文案，保证功能链路可跑通；也可本地起 Ollama 后把 `LLM_BASE_URL` 指向 `http://127.0.0.1:11434/v1`。
</details>

<details>
<summary><b>数据集太大（2.2GB）怎么管理？</b></summary>

`dataset/` 已在 `.gitignore` 中排除，通过网盘 / GitHub Release 分发。协作时只需把文件放进 `dataset/` 目录即可，路径由 `.env` 的 `DATA_DIR` 控制。
</details>

---

## 八、开发路线图

- [x] 项目定位与技术栈选型
- [x] 数据集探查与清洗口径确认
- [ ] 阶段 1：数据预处理与序列数据集构建（15%）
- [ ] 阶段 2：核心算法研发与对比实验（35%）
- [ ] 阶段 3：全栈系统开发与联调（35%）
- [ ] 阶段 4：系统测试与论文撰写（15%）

---

## 九、团队与联系方式

| 角色 | 姓名 | 负责模块 | 邮箱 |
|---|---|---|---|
| 项目负责人 / 算法 | *待填* | 推荐算法、多智能体编排 | *待填* |
| 后端开发 | *待填* | FastAPI 服务、数据库、定时任务 | *待填* |
| 前端开发 | *待填* | Vue 3 页面、ECharts 可视化 | *待填* |
| 指导教师 | *待填* | 选题指导与进度把控 | *待填* |

- 问题反馈：请提 [Issue](https://github.com/<your-name>/anirec/issues)
- 项目文档：`docs/` 目录

---

## License

本项目采用 MIT 协议开源，仅用于学习与毕业设计用途。数据版权归 MyAnimeList 所有。

<div align="center">
<sub>If this project helps you, please give it a ⭐</sub>
</div>

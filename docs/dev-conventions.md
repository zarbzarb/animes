# 开发规范（Development Conventions）

> 本文是团队协作的「共同语言」：**分支怎么建、提交怎么写、代码怎么放、Review 看什么、文档怎么维护**。
> 目的是让三个人写出像一个人写的代码，也让答辩时提交记录本身就是工程能力的证据。

---

## 一、Git 工作流

### 1.1 分支模型（简化版 Git Flow）

```
main          ──●────────────●────────────●──→   永远可运行、可演示
                 ╲          ╱ ╲          ╱
develop       ────●───●────●───●───●────●───→   集成分支，功能在此汇合
                   ╲ ╱        ╲ ╱
feature/xxx    ─────●          ●                功能开发
fix/xxx        ────────────────●                缺陷修复
```

| 分支 | 用途 | 来源 | 合入 |
|---|---|---|---|
| `main` | 可演示的稳定版本 | — | 只从 `develop` 合入，且必须是里程碑节点 |
| `develop` | 日常集成 | `main` | 功能分支合入 |
| `feature/<模块>-<简述>` | 新功能 | `develop` | → `develop`（PR + Review） |
| `fix/<简述>` | 修 bug | `develop` | → `develop` |
| `hotfix/<简述>` | 线上紧急修复 | `main` | → `main` + `develop` |
| `exp/<实验名>` | 实验性代码（可能被丢弃） | `develop` | 不合并，仅作记录 |

**分支命名示例**：

```
feature/recall-multi-interest
feature/frontend-interest-radar
feature/agent-explain-prompt
fix/sasrec-mask-leak
fix/redis-cache-stale
exp/k-search-2-4-6-8
```

### 1.2 分支保护规则

| 分支 | 规则 |
|---|---|
| `main` | 禁止直接 push；必须 PR；必须 1 人 Review；CI 必须通过 |
| `develop` | 禁止直接 push（除紧急情况）；PR + CI 通过 |
| `feature/*` | 自由 push |

### 1.3 日常流程

```bash
# 1) 从 develop 拉新分支
git checkout develop && git pull
git checkout -b feature/recall-multi-interest

# 2) 开发（小步提交，见第二节）
git add agents/recall models/multi_interest
git commit -m "feat(recall): 实现多兴趣胶囊召回，K=4"
git commit -m "test(recall): 补充胶囊召回单元测试"

# 3) 同步 develop（避免大冲突）
git fetch origin && git rebase origin/develop

# 4) 推送并开 PR
git push -u origin feature/recall-multi-interest
# → 在 GitHub 开 PR，指定 Reviewer

# 5) Review 通过后 squash 合并到 develop
git checkout develop && git pull
git merge --squash feature/recall-multi-interest
git commit -m "feat(recall): 多兴趣胶囊召回 (#12)"

# 6) 删分支
git branch -d feature/recall-multi-interest
git push origin --delete feature/recall-multi-interest
```

### 1.4 绝对禁止的操作

| ❌ 禁止 | 原因 | 正确做法 |
|---|---|---|
| `git push --force` 到 `main`/`develop` | 会覆盖他人提交 | 用 `--force-with-lease` 只在自己的 `feature/*` 分支 |
| 提交 `.env` / 模型权重 / 数据集 | 体积大 + 泄密 | 已在 `.gitignore` 排除，确认后再提交 |
| 一个 commit 混多个不相关改动 | 无法回滚、Review 困难 | 拆成多个原子提交 |
| 直接改 `main` | 破坏稳定性 | 走 PR |
| `git commit -m "update"` | 无信息量 | 用 Conventional Commits |

---

## 二、提交规范（Conventional Commits）

### 2.1 格式

```
<type>(<scope>): <subject>

<body 可选>

<footer 可选>
```

### 2.2 type 清单

| type | 用途 | 示例 |
|---|---|---|
| `feat` | 新功能 | `feat(recall): 实现多兴趣胶囊召回` |
| `fix` | 修 bug | `fix(model): 修正注意力掩码导致的未来信息泄漏` |
| `docs` | 文档 | `docs(arch): 补充 Agent 层与算法层的数据契约` |
| `style` | 格式（不影响逻辑） | `style: 统一使用 ruff format` |
| `refactor` | 重构 | `refactor(fusion): 抽出打分归一化函数` |
| `perf` | 性能优化 | `perf(recall): 批量前向 batch 从 128 提到 512` |
| `test` | 测试 | `test(metrics): 补充 NDCG 与 sklearn 对齐用例` |
| `chore` | 构建/依赖/配置 | `chore(deps): 升级 torch 到 2.4` |
| `exp` | 实验相关 | `exp(ablation): 记录 K=6 的消融结果` |
| `revert` | 回滚 | `revert: 回滚 #45（引入召回重复问题）` |

### 2.3 scope 清单（与本项目目录对应）

`orchestrator` `profile` `recall` `coldstart` `fusion` `explain` `drift` `chat` `dataops` `eval`
`sasrec` `multi-interest` `content` `metrics` `api` `db` `frontend` `arch` `deps` `ci`

### 2.4 subject 要求

| 要求 | 好 | 坏 |
|---|---|---|
| 用祈使句，不加句号 | `实现多兴趣胶囊召回` | `实现了多兴趣胶囊召回了。` |
| 不超过 50 字 | `修正注意力掩码的未来信息泄漏` | `修正了一个在训练过程中因为掩码构造顺序问题导致模型看到了未来信息从而指标虚高的问题` |
| 说清"做了什么" | `冻结 SASRec 前两层只训胶囊网络` | `改了点东西` |
| 中文（本项目统一） | `补充冷启动子集构造脚本` | `add cold start subset script` |

### 2.5 完整示例

```
feat(recall): 实现多兴趣胶囊召回，K=4

- models/multi_interest/capsule.py: 初级胶囊 + 3 轮动态路由
- agents/recall/strategy.py: 每胶囊 TopK 召回后合并去重
- 支持过滤已看物品，过滤率约 12%

实验影响: 消融实验 E2-2 需重新跑
Refs: #8
```

### 2.6 commit 与实验的关联

**改动了影响实验结果的代码，必须在 commit body 里写 `实验影响:` 一行。** 这样后续能通过 `git log --grep="实验影响"` 找出所有需要重跑实验的提交。

---

## 三、代码规范

### 3.0 路径与工作目录（硬约束）

**所有 `scripts/` 下的脚本必须与「当前工作目录」无关**，在任何目录下启动结果一致。

- 配置里的路径一律写作**相对项目根**的形式（`./dataset/animes.csv`、`./data/processed`）。
- 脚本读配置后必须把这些相对路径**锚定到项目根**，再交给 `open` / `pandas` / `numpy`。
  参考实现：`scripts/preprocess.py` 的 `anchor_paths()` + `load_cfg()`。
- 只有**真正是路径**的字段才锚定。`output.sequence_file`、`content_vec_file`
  这类是**文件名**，之后还要与 `processed_dir` / `feature_dir` 做 `join`，
  锚定会破坏它们。
- 脚本内部定位项目根统一用
  `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`，
  禁止依赖 `os.getcwd()` 或裸相对路径。
- 命令行参数（`--config` / `--out` 等）按同一约定解释：相对路径视为相对项目根。

**反例（阶段一真实踩过的坑）**：`build_anime_meta()` 曾把 `cfg["input"]["anime_meta"]`
直接交给 `pd.read_csv`，于是

```bash
cd F:/pj && python scripts/preprocess.py          # ✅ 能跑
python F:/pj/scripts/preprocess.py                # ❌ FileNotFoundError: './dataset/animes.csv'
```

只有在项目根目录下才碰巧能跑 —— 这类 bug 在 CI、IDE「运行」按钮、
任务计划里都会复现。新增脚本请直接用 `anchor_paths()`。

### 3.1 Python

**格式化与检查工具**：`ruff`（lint + format）+ `black`（备选）+ `pre-commit`

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM", "RUF"]
ignore = ["E501"]        # 行长由 formatter 处理

[tool.ruff.lint.isort]
known-first-party = ["agents", "models", "server"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=server --cov=agents --cov=models --cov-report=term-missing"
```

**命名规范**：

| 对象 | 规范 | 示例 |
|---|---|---|
| 模块 | 小写下划线 | `recommend_service.py` |
| 类 | 大驼峰 | `FusionRankAgent` |
| 函数/变量 | 小写下划线 | `merge_candidates` |
| 常量 | 全大写下划线 | `MAX_SEQ_LEN` |
| 私有 | 前置单下划线 | `_normalize_scores` |
| 类型别名 | 大驼峰 | `AnimeId = int` |

**类型注解**：公开函数必须有完整类型注解。

```python
# ✅ 好
async def recall(self, payload: RecallInput) -> RecallOutput:
    ...

# ❌ 差
async def recall(self, payload):
    ...
```

**文档字符串**：公开类与函数必须有 docstring，说明**为什么**而不只是**是什么**。

```python
def fusion_score(behavior: Tensor, content: Tensor, n_inter: Tensor) -> Tensor:
    """融合行为分与内容分。

    冷启动物品（n_inter < threshold）提高内容分权重至 0.5，
    因为行为分对无交互物品不可靠，而内容向量仍能提供有效信号。

    Args:
        behavior: 行为路得分 [B, N]
        content:  内容路得分 [B, N]
        n_inter:  候选物品的交互数 [N]，用于判定冷启动

    Returns:
        融合后的最终得分 [B, N]
    """
```

**异常处理**：禁止裸 `except`。

```python
# ❌ 禁止
try:
    result = model.encode(seq)
except:
    pass

# ✅ 正确：捕获具体异常 + 记录 + 有明确降级
try:
    result = model.encode(seq)
except torch.cuda.OutOfMemoryError as e:
    logger.warning("模型前向 OOM，降级到 ItemCF: %s", e)
    return self._fallback_itemcf(user_id)
except ModelNotLoadedError:
    raise   # 这是配置错误，必须暴露
```

**日志**：用 `logging`，禁止 `print`（脚本除外）。

```python
# ✅
logger.info("[%s][A2] recall done candidates=%d elapsed=%dms", trace_id, n, ms)

# ❌
print(f"召回完成 {n} 条")
```

### 3.2 分层约束（静态检查）

```bash
# CI 中执行，违反则构建失败
python scripts/check_imports.py
```

| 规则 | 检查内容 |
|---|---|
| R1 | `models/` 不得 import `agents/` 或 `server/` |
| R2 | `models/` 不得 import `torch` 之外的重依赖（如 `redis`、`sqlalchemy`） |
| R3 | `agents/` 不得 import `server/` |
| R4 | `server/api/` 不得 import `models/`（必须经 service / adapter） |
| R5 | 任何目录不得 import `tests/` |

### 3.3 Vue / TypeScript

| 项 | 规范 |
|---|---|
| 组件名 | 大驼峰，`.vue` 单文件组件 |
| 组合式 API | 统一用 `<script setup lang="ts">`，不用 Options API |
| Props | 用 `defineProps<T>()` 泛型写法 |
| 状态 | 跨页面用 Pinia，页面内用 `ref` / `reactive` |
| 请求 | 全部走 `src/api/`，禁止组件里直接 `axios` |
| 样式 | 优先 `<style scoped>`；用 CSS 变量做主题 |
| 类型 | 接口返回类型在 `src/types/` 统一定义，禁止 `any` |

```vue
<script setup lang="ts">
import type { RecItem } from '@/types/recommend'

const props = defineProps<{ item: RecItem }>()
const emit = defineEmits<{ click: [animeId: number] }>()
</script>
```

### 3.4 SQL / 数据库

| 规则 | 说明 |
|---|---|
| 禁止 `SELECT *` | 显式列名 |
| 必须走 ORM | 除 `scripts/` 中的批量导入，业务代码用 SQLAlchemy |
| 索引先行 | 新增查询前先看 `EXPLAIN`，确认走索引 |
| 迁移必须写 Alembic | 不允许手改线上表结构 |
| 大表操作分批 | `recommend_result`、`agent_trace` 的删除必须 LIMIT 分批 |

```python
# ✅ 分批删除，避免长事务锁表
while True:
    n = db.execute(
        "DELETE FROM agent_trace WHERE created_at < :d LIMIT 5000", {"d": cutoff}
    ).rowcount
    if n == 0:
        break
    db.commit()
```

### 3.5 Agent 代码专用规范

| 规则 | 说明 |
|---|---|
| 必须继承 `BaseAgent` | 统一走 `handle()`，不要自己写入口 |
| 必须定义 `schemas.py` | 输入输出用 Pydantic，禁止裸 dict |
| 必须声明优先级与超时 | 在 `config.py` 中定义，不要硬编码 |
| 必须有降级路径 | `invoke()` 内任何外部调用都要有 fallback |
| 不得跨 Agent 直接调用 | 通过 registry 或 A0 调度 |
| LLM 调用必须走 `llm.py` | 统一超时、重试、计量、脱敏 |
| 每个 Agent 必须有 README.md | 说明职责/输入输出/边界/降级 |

---

## 四、测试规范

### 4.1 测试分层

| 层 | 目录 | 内容 | 覆盖率目标 |
|---|---|---|---|
| 单元测试 | `tests/test_models/` `tests/test_metrics/` | 模型前向形状、指标正确性 | 70% |
| Agent 测试 | `tests/test_agents/` | 契约、降级、幂等、错误码 | 60% |
| 接口测试 | `tests/test_api/` | 请求响应、鉴权、错误码 | 60% |
| Prompt 测试 | `tests/prompts/` | 分类准确率、Schema 通过率、幻觉率 | — |
| 集成测试 | `tests/test_integration/` | 端到端链路（含降级场景） | — |

### 4.2 必测清单

| 必须测的内容 | 原因 |
|---|---|
| `models/eval/metrics.py` 的 HR/NDCG | 指标错了整个实验全错，用 sklearn/手工算例对齐 |
| Agent 的降级路径 | 降级逻辑最容易在重构中失效 |
| Agent 的 Pydantic Schema | 契约变更会静默影响上下游 |
| 注意力掩码无未来信息泄漏 | 序列推荐的经典 bug，会导致指标虚高 |
| 融合权重归一 | 配错会导致打分失真 |
| 接口鉴权与越权 | 用户 A 不能读用户 B 的追番 |

### 4.3 关键测试示例

```python
# tests/test_metrics/test_ndcg.py
def test_ndcg_hand_calculation():
    """与手工计算对齐：正样本排第 2 位时 NDCG@5 = 1/log2(3)"""
    ranks = [2]           # 正样本排名（1-based）
    expected = 1.0 / math.log2(3)
    assert abs(ndcg_at_k(ranks, k=5) - expected) < 1e-9

def test_ndcg_perfect_ranking():
    """正样本排第 1 位时应为 1.0"""
    assert abs(ndcg_at_k([1], k=10) - 1.0) < 1e-9

def test_hr_at_k_boundary():
    assert hr_at_k([10], k=10) == 1.0
    assert hr_at_k([11], k=10) == 0.0
```

```python
# tests/test_models/test_no_future_leak.py
def test_causal_mask_no_future_leak():
    """改动序列末位物品不应影响前面位置的输出（因果性验证）"""
    seq = torch.randint(1, 100, (1, 10))
    out1 = model.encode(seq)[0]
    seq2 = seq.clone(); seq2[0, -1] = 99
    out2 = model.encode(seq2)[0]
    # 前 9 个位置的输出必须完全一致
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)
```

```python
# tests/test_agents/test_recall_degrade.py
async def test_recall_falls_back_to_itemcf_when_model_missing(monkeypatch):
    monkeypatch.setattr(adapter, "encode", raise_model_unavailable)
    out = await RecallAgent().invoke({"user_id": 1024, "user_seq": [1, 2, 3]})
    assert out["meta"]["degraded"] is True
    assert out["meta"]["degraded_reason"] == "fallback_to_itemcf_hot"
    assert len(out["candidates"]) > 0
```

```python
# tests/test_api/test_records_permission.py
def test_cannot_modify_others_record(client, user_a_token, user_b_record):
    r = client.put(f"/api/v1/records/{user_b_record.id}",
                   json={"status": 2}, headers={"Authorization": f"Bearer {user_a_token}"})
    assert r.status_code == 403
    assert r.json()["code"] == 40301
```

### 4.4 运行测试

```bash
pytest tests/ -v                          # 全部
pytest tests/test_metrics/ -v             # 只测指标
pytest -k "degrade" -v                    # 只测降级
pytest --cov=server --cov=agents --cov-report=html   # 覆盖率报告

# Prompt 回归（需要 LLM Key，耗时较长，单独跑）
pytest tests/prompts/ -v --run-llm
```

---

## 五、Code Review 规范

### 5.1 作者自查清单（提交 PR 前）

- [ ] 本地 `ruff check .` 与 `ruff format --check .` 通过
- [ ] `pytest tests/` 全部通过
- [ ] `python scripts/check_imports.py` 通过（无跨层依赖）
- [ ] 新增功能有对应测试
- [ ] 变更涉及文档的部分已同步更新 `docs/`
- [ ] 无 `print` / 无调试代码 / 无注释掉的死代码
- [ ] 无密钥、无大文件、无 `.env`
- [ ] commit message 符合 Conventional Commits
- [ ] 若改动影响实验结果，body 中已标注 `实验影响:`

### 5.2 Reviewer 检查重点

| 检查项 | 关注点 |
|---|---|
| **分层正确性** | 有没有把算法逻辑写进 Agent？有没有把 SQL 写进模型？ |
| **契约一致性** | Schema 变更是否影响上下游？是否更新了文档？ |
| **降级完整性** | 新增外部调用是否都有失败路径？ |
| **指标正确性** | 有没有重写指标计算？有没有绕过 `evaluator.py`？ |
| **实验公平性** | 改了超参有没有说明？是否与其他配置可比？ |
| **N+1 查询** | 循环里查数据库了吗？ |
| **大表风险** | 有没有全表扫描 / 无 LIMIT 的删除？ |
| **命名与可读性** | 半年后自己还看得懂吗？ |

### 5.3 Review 用语约定

| 前缀 | 含义 | 是否阻塞合并 |
|---|---|---|
| `[blocker]` | 必须改，否则不能合 | ✅ 阻塞 |
| `[suggestion]` | 建议改，作者可判断 | ❌ 不阻塞 |
| `[question]` | 有疑问，需作者解释 | 🔶 澄清后可合 |
| `[nit]` | 吹毛求疵的小问题 | ❌ 不阻塞 |
| `[praise]` | 写得好的地方，明确说出来 | — |

---

## 六、文档规范

### 6.1 文档清单与责任人

| 文档 | 更新时机 | 责任人 |
|---|---|---|
| `README.md` | 功能/启动方式变化 | 项目负责人 |
| `docs/architecture.md` | 分层/数据流/ADR 变化 | 项目负责人 |
| `docs/agents.md` | 新增/修改 Agent | Agent 开发者 |
| `docs/agent-interaction-protocol.md` | 消息格式/错误码/降级变化 | Agent 开发者 |
| `docs/agent-prompt-design.md` | **任何 Prompt 改动** | Prompt 作者 |
| `docs/agent-memory-design.md` | 记忆机制变化 | 画像 Agent 负责人 |
| `docs/project-structure.md` | **新增目录或顶层文件** | 提交者 |
| `docs/database-design.md` | **任何表结构变化** | 后端负责人 |
| `docs/api-specification.md` | **任何接口变化** | 后端负责人 |
| `docs/evaluation-plan.md` | 实验口径变化 | 算法负责人 |
| `docs/ablation-study.md` / `result-analysis.md` | 实验结果产出 | 算法负责人 |
| `docs/config-guide.md` | 配置项增删 | 提交者 |
| `docs/dev-conventions.md` | 规范调整 | 全队讨论 |

### 6.2 铁律

> **代码与文档必须同一个 PR。没有例外。**
> Reviewer 看到代码改了但文档没改，直接 `[blocker]`。

### 6.3 文档写作要求

| 要求 | 说明 |
|---|---|
| 中文为主 | 术语保留英文原文，如「多兴趣胶囊（multi-interest capsule）」 |
| 结论前置 | 每节开头先给结论，再展开论证 |
| 用图表 | 流程用 Mermaid，对比用表格，避免大段文字 |
| 标注来源 | 出现的数字必须注明出处（文件名/脚本名/表名） |
| 占位统一 | `TODO(填写项)` / `待填` / `⚠️ 占位`，便于全局搜索 |
| 禁止编造 | **未实测的指标数字一律不得填写**，宁可留占位 |

### 6.4 文档目录一致性

新增目录时同步更新三处：
1. `docs/project-structure.md` 的目录树
2. `README.md` 的目录结构说明
3. 该目录下的 `README.md`（如果是 Agent 或 model 子包）

---

## 七、提交前自动化（pre-commit）

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=2048]        # 禁止提交 > 2MB 文件（防误传权重）
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks                # 密钥泄漏扫描

  - repo: local
    hooks:
      - id: check-imports
        name: 检查分层依赖方向
        entry: python scripts/check_imports.py
        language: system
        pass_filenames: false
```

**安装**：

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files      # 手动全量跑一次
```

---

## 八、CI 流水线

```yaml
# .github/workflows/ci.yml（关键步骤）
name: CI
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - name: Lint
        run: ruff check . && ruff format --check .
      - name: 分层依赖检查
        run: python scripts/check_imports.py
      - name: 单元测试
        run: pytest tests/test_metrics tests/test_models tests/test_agents -v
      - name: 接口测试
        run: pytest tests/test_api -v
```

**CI 失败一律不允许合并**（包含 Lint、分层检查、测试三项）。

---

## 九、协作沟通规范

| 场景 | 规范 |
|---|---|
| 任务分配 | 用 GitHub Issue，必须写清：目标、验收标准、涉及模块 |
| 进度同步 | 每周一次简会：上周完成 / 本周计划 / 阻塞项 |
| 阻塞上报 | 阻塞超过半天必须说出来，不要闷头卡住 |
| 接口变更 | **先沟通再改**，接口是契约，不能单方面改 |
| 表结构变更 | 同上，必须先同步后端与算法两边 |
| 实验结论 | 不准说"效果挺好的"，要给数字与出处 |
| 代码求助 | 说清：想做什么、试了什么、报什么错、期望什么 |

### Issue 模板

```markdown
## 目标
实现多兴趣胶囊召回，K=4

## 验收标准
- [ ] `agents/recall/` 返回每个候选的 `interest_id`
- [ ] 已看物品被过滤
- [ ] 有单元测试覆盖降级路径
- [ ] 单用户召回耗时 < 50ms（GPU）

## 涉及模块
`agents/recall/` `models/multi_interest/`

## 依赖
依赖 #8（胶囊网络实现完成）

## 备注
消融实验 E2-2 需要在此完成后重跑
```

---

## 十、答辩前的规范化准备

| 项 | 准备内容 |
|---|---|
| 提交记录 | 检查 `git log --oneline` 是否清晰、无 "update" 类无意义提交 |
| 分支清理 | 删除已合并的 `feature/*`，保留 `main` / `develop` |
| 文档完整性 | 按 `docs/README.md` 清单逐项确认无 `待填`（除实验结果） |
| 无敏感信息 | `gitleaks detect --source .` 无告警 |
| 环境可复现 | 按 `README.md` 快速启动步骤在干净环境验证一遍 |
| 演示数据 | 预置演示账号与追番记录，保证演示页面有内容 |
| 离线指标 | 至少跑完 E1 + E2，图表已生成 |
| 降级演练 | 演示时可现场停掉 LLM，展示模板兜底与 `meta.degraded` |

---

## 十一、相关文档

- 目录结构 → [project-structure.md](project-structure.md)
- 配置管理 → [config-guide.md](config-guide.md)
- 文档索引 → [README.md](README.md)
- 接口契约 → [api-specification.md](api-specification.md)

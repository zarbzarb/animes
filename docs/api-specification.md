# 接口规范（API Specification）

> **Base URL**：`http://127.0.0.1:8000/api/v1`　　**协议**：HTTP/JSON（UTF-8）+ SSE（流式）
> **自动文档**：启动后访问 `/docs`（Swagger UI）或 `/redoc`。本文与代码必须同步。
> 数据表对应见 [database-design.md](database-design.md)，错误码含义见 [agent-interaction-protocol.md](agent-interaction-protocol.md) 第四节。

---

## 一、通用约定

### 1.1 统一响应体

**所有**接口（含错误）都返回如下结构：

```jsonc
{
  "code": 0,                    // 0=成功；非 0 见错误码表
  "message": "ok",              // 人类可读信息
  "data": { },                  // 业务载荷；错误时为 null
  "trace_id": "tr_7f3a9c21e0b4",// 全链路追踪ID，排查问题用
  "elapsed_ms": 265             // 服务端处理耗时
}
```

### 1.2 分页约定

请求：`?page=1&size=20`（`page` 从 1 开始，`size` 默认 20、最大 100）

响应 `data`：

```jsonc
{
  "list": [ ],
  "pagination": {
    "page": 1, "size": 20, "total": 168, "pages": 9
  }
}
```

### 1.3 鉴权

除 `/auth/*` 与 `/animes/*`（只读）外，所有接口需在 Header 带：

```
Authorization: Bearer <access_token>
```

| 项 | 说明 |
|---|---|
| TTL | 1440 分钟（`JWT_EXPIRE_MINUTES`） |
| 刷新 | `POST /auth/refresh` |
| 失败 | `401` + `code=40101`，前端跳登录 |

### 1.4 通用错误响应示例

```jsonc
{
  "code": 40101,
  "message": "登录状态已失效，请重新登录",
  "data": null,
  "trace_id": "tr_7f3a9c21e0b4",
  "elapsed_ms": 3
}
```

### 1.5 幂等

- 写接口（`POST` / `PUT` / `DELETE`）支持可选 Header `Idempotency-Key`。
- 服务端以该键写入 Redis `idem:api:{key}`（TTL 10min），重复请求直接返回首次结果。

---

## 二、接口总览

| # | 模块 | 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|---|---|
| 1 | 认证 | POST | `/auth/register` | ❌ | 注册 |
| 2 | 认证 | POST | `/auth/login` | ❌ | 登录（带验证码） |
| 2b | 认证 | GET | `/auth/captcha` | ❌ | 登录验证码（SVG，一次一密） |
| 3 | 认证 | POST | `/auth/refresh` | ✅ | 刷新 token |
| 4 | 认证 | POST | `/auth/logout` | ✅ | 登出 |
| 5 | 用户 | GET | `/users/me` | ✅ | 当前用户信息 |
| 6 | 用户 | PUT | `/users/me` | ✅ | 修改资料 |
| 7 | 用户 | PUT | `/users/me/password` | ✅ | 改密码 |
| 8 | 用户 | DELETE | `/users/me/memory` | ✅ | 清空个人记忆数据 |
| 9 | 动漫 | GET | `/animes` | ❌ | 动漫列表（搜索/筛选/分页） |
| 10 | 动漫 | GET | `/animes/{id}` | ❌ | 动漫详情 |
| 11 | 动漫 | GET | `/animes/{id}/similar` | ❌ | 相似动漫 |
| 12 | 动漫 | GET | `/genres` | ❌ | 12 类题材列表 |
| 13 | 追番 | GET | `/records` | ✅ | 我的追番列表 |
| 14 | 追番 | POST | `/records` | ✅ | 新增/更新追番记录 |
| 15 | 追番 | PUT | `/records/{id}` | ✅ | 修改追番记录 |
| 16 | 追番 | DELETE | `/records/{id}` | ✅ | 删除追番记录 |
| 17 | 追番 | GET | `/records/timeline` | ✅ | 观看时间轴 |
| 18 | 追番 | GET | `/records/stats` | ✅ | 追番统计概览 |
| 19 | **推荐** | GET | `/recommend/feed` | ✅ | **综合推荐 Top20** |
| 20 | **推荐** | GET | `/recommend/by-genre` | ✅ | **分类推荐** |
| 21 | **推荐** | GET | `/recommend/new-anime` | ✅ | **新番专属推荐** |
| 22 | **推荐** | GET | `/recommend/{anime_id}/explain` | ✅ | **推荐解释** |
| 23 | **推荐** | POST | `/recommend/feedback` | ✅ | 推荐反馈（曝光/点击/不感兴趣） |
| 24 | 对话 | POST | `/chat/message` | ✅ | 对话推荐（SSE 流式） |
| 25 | 对话 | POST | `/chat/session` | ✅ | 新建会话 |
| 26 | 对话 | GET | `/chat/history` | ✅ | 会话历史 |
| 27 | 分析 | GET | `/analysis/interest-radar` | ✅ | 兴趣雷达图 |
| 28 | 分析 | GET | `/analysis/drift-trend` | ✅ | 题材漂移趋势 |
| 29 | 分析 | GET | `/analysis/drift-points` | ✅ | 漂移点标注 |
| 30 | 分析 | POST | `/analysis/agent-invoke` | ✅ | 直接调 Agent（调试用） |
| 30b | 分析 | GET | `/analysis/profile-summary` | ✅ | 画像/状态/评分分布/月度节奏汇总 |
| 30c | 用户 | POST | `/users/me/avatar` | ✅ | 头像上传（jpg/png/webp/gif ≤2MB） |
| 31 | 管理 | GET | `/admin/animes` | 🔒 | 动漫管理列表 |
| 32 | 管理 | POST | `/admin/animes` | 🔒 | 新增动漫 |
| 33 | 管理 | PUT | `/admin/animes/{id}` | 🔒 | 编辑动漫 |
| 34 | 管理 | PUT | `/admin/animes/{id}/offline` | 🔒 | 下架动漫 |
| 35 | 管理 | GET | `/admin/users` | 🔒 | 用户列表 |
| 36 | 管理 | GET | `/admin/metrics` | 🔒 | 推荐效果监控 |
| 37 | 管理 | GET | `/admin/cold-start` | 🔒 | 冷启动看板 |
| 38 | 管理 | GET | `/admin/agents/health` | 🔒 | Agent 健康状态 |
| 39 | 管理 | GET | `/admin/traces/{trace_id}` | 🔒 | 调用链详情 |
| 40 | 管理 | POST | `/admin/tasks/offline-recommend` | 🔒 | 手动触发离线推荐 |
| 41 | 内部 | POST | `/internal/agents/{name}` | 🛡️ | Agent 内部调用（内网） |

🔒 = 需管理员角色　🛡️ = 内网 IP + 内部 token

---

## 三、核心接口详解

### 3.1 认证模块

#### `POST /auth/register`

**请求**

```jsonc
{ "username": "anifan", "password": "P@ssw0rd", "email": "a@b.com", "nickname": "番剧迷" }
```

**校验规则**

| 字段 | 规则 |
|---|---|
| `username` | 4-50 字符，字母数字下划线，唯一 |
| `password` | 8-64 字符，至少含字母与数字 |
| `email` | 可选，合法邮箱，唯一 |
| `nickname` | 1-50 字符，默认取 username |

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_a1b2c3d4e5f6", "elapsed_ms": 42,
  "data": {
    "user": { "id": 1024, "username": "anifan", "nickname": "番剧迷", "role": 0, "avatar_url": null },
    "access_token": "eyJhbGciOi...",
    "token_type": "Bearer",
    "expires_in": 86400
  }
}
```

| 错误码 | 场景 |
|---|---|
| `40001` | 参数校验失败 |
| `40901` | 用户名/邮箱已存在 |

---

#### `GET /auth/captcha`

**响应**

```jsonc
{
  "code": 0, "message": "ok",
  "data": {
    "captcha_id": "c2218a199769...",   // 登录时原样带回
    "svg": "<svg .../>",               // 服务端渲染的扭曲字符图（4 位，浏览器直接内联）
    "expires_in": 300                  // 秒
  }
}
```

> 一次一密：答案只存服务端缓存（TTL 5 分钟），**无论校验对错都立即作废**；
> 大小写不敏感；字符集已去掉 `0O1lI` 等易混字形。生产环境置
> `AUTH_CAPTCHA_STRICT=true` 后，登录不带验证码直接 `40001`。

---

#### `POST /auth/login`

**请求**

```jsonc
{ "username": "anifan", "password": "P@ssw0rd",
  "captcha_id": "c2218a199769...", "captcha_code": "a7k2" }
```

`captcha_id` / `captcha_code` 由 `GET /auth/captcha` 获得；默认宽松
（不带也放行，供冒烟/脚本），strict 模式下必填。
```

**响应**：同注册（不含 `user` 时也返回概要）。

| 错误码 | 场景 |
|---|---|
| `40101` | 用户名或密码错误 |
| `40301` | 账号被禁用 |

---

### 3.2 追番管理

#### `GET /records`

**Query**

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `status` | int | — | 0想看 1在看 2已看 3弃番，不传=全部 |
| `genre_id` | int | — | 按题材筛选 |
| `sort` | string | `updated_desc` | `updated_desc` / `rating_desc` / `watched_desc` |
| `page` / `size` | int | 1 / 20 | 分页 |

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 18,
  "data": {
    "list": [
      {
        "id": 55231,
        "anime": {
          "id": 1201, "src_anime_id": 16498, "title": "进击的巨人",
          "type": "TV", "year": 2013, "score": 8.55, "episodes": 25,
          "image_url": "https://...", "genres": ["热血战斗", "剧情文艺"]
        },
        "status": 2, "status_label": "已看", "rating": 9, "progress": 25,
        "watched_at": "2026-08-14", "updated_at": "2026-08-14T21:03:00Z"
      }
    ],
    "pagination": { "page": 1, "size": 20, "total": 168, "pages": 9 }
  }
}
```

---

#### `POST /records`

> **重要**：本接口会在写入成功后**异步**触发增量重排（A1 → A2 → A4），并让 `rec:{uid}` 缓存失效。响应**不等待**重排完成。

**请求**

```jsonc
{
  "anime_id": 1201,
  "status": 1,          // 0想看 1在看 2已看 3弃番
  "rating": 9,          // 可选，1-10
  "progress": 12,       // 可选
  "watched_at": "2026-09-10"  // 可选
}
```

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 25,
  "data": {
    "record_id": 55231,
    "is_new": true,
    "recompute_scheduled": true,     // 是否已投递增量重排任务
    "recompute_eta_ms": 800
  }
}
```

| 错误码 | 场景 |
|---|---|
| `40401` | `anime_id` 不存在或已下架 |
| `40901` | 重复提交（幂等键冲突） |

---

#### `GET /records/timeline`

**Query**：`granularity=month|quarter|year`（默认 `month`）、`limit`

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 22,
  "data": {
    "timeline": [
      {
        "period": "2026-07",
        "count": 7,
        "genres": [{ "genre": "悬疑推理", "count": 4 }, { "genre": "日常治愈", "count": 3 }],
        "items": [
          { "anime_id": 1201, "title": "进击的巨人", "rating": 9, "watched_at": "2026-07-03" }
        ]
      }
    ]
  }
}
```

---

### 3.3 推荐模块（核心）

#### `GET /recommend/feed` —— 综合推荐

**Query**

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `size` | int | 20 | 返回条数，1-20 |
| `with_explain` | bool | true | 是否附带解释理由 |
| `refresh` | bool | false | 强制绕过缓存重算 |

**内部协作链路**：`A0 → A1 → (A2 ∥ A3) → A4 → A5`
命中缓存时直接返回，耗时 ~15ms；未命中 ~265ms。

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_7f3a9c21e0b4", "elapsed_ms": 265,
  "data": {
    "items": [
      {
        "rank": 1,
        "anime": {
          "id": 1407, "src_anime_id": 40748, "title": "咒术回战",
          "type": "TV", "year": 2020, "score": 8.62, "episodes": 24,
          "image_url": "https://...", "genres": ["热血战斗", "超自然灵异"]
        },
        "final_score": 0.9124,
        "behavior_score": 0.9310,
        "content_score": 0.7200,
        "interest_id": 0,
        "interest_label": "热血战斗",
        "is_cold_start": false,
        "explain": {
          "reason": "你近期观看了《进击的巨人》《鬼灭之刃》等热血战斗番，本作同属该题材，题材匹配度 87%。",
          "core_items": [
            { "title": "进击的巨人", "weight": 0.31 },
            { "title": "鬼灭之刃", "weight": 0.24 },
            { "title": "一拳超人", "weight": 0.18 }
          ],
          "match_percent": 87,
          "matched_genres": ["热血战斗"],
          "source": "llm"        // llm | template
        }
      }
    ],
    "meta": {
      "cache_hit": false,
      "profile_version": 37,
      "agent_chain": ["A1", "A2", "A3", "A4", "A5"],
      "degraded": [],
      "batch_id": "20260912_0300"
    }
  }
}
```

**降级表现**

| 情况 | `meta.degraded` | 用户可见变化 |
|---|---|---|
| A3 失败 | `["A3"]` | 无新番候选，结果仍正常 |
| A5 超时 | `["A5"]` | `explain.source = "template"` |
| A2 失败 | `["A2"]` | 结果来自 ItemCF 热门榜，`meta.fallback = "itemcf_hot"` |
| 缓存不可用 | `[]` | `cache_hit=false`，耗时上升 |

---

#### `GET /recommend/by-genre` —— 分类推荐

**Query**：`genre_id`（必填，1-12）、`size`（默认 20）

**响应**：结构同 `/recommend/feed`；`data.items[].explain.matched_genres` 必定包含该 `genre_id` 对应题材。

| 错误码 | 场景 |
|---|---|
| `40001` | `genre_id` 超出 1-12 |

---

#### `GET /recommend/new-anime` —— 新番专属推荐

**Query**：`season`（如 `2026Q3`，默认当前季度）、`size`（默认 20）

**内部链路**：`A0 → A1 → A3 → A4 → A5`（**跳过 A2**，纯内容召回）

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 190,
  "data": {
    "season": "2026Q3",
    "items": [
      {
        "rank": 1,
        "anime": { "id": 20301, "title": "新番A", "year": 2026, "genres": ["日常治愈"], "image_url": "https://..." },
        "final_score": 0.7812,
        "content_score": 0.8300,
        "behavior_score": 0.0,
        "is_cold_start": true,
        "cold_start_badge": "冷启动推荐",
        "explain": {
          "reason": "这是本季新番，暂无观看数据，但与你看过的《夏目友人帐》同属日常治愈题材，题材匹配度 78%。",
          "core_items": [{ "title": "夏目友人帐", "weight": 0.29 }],
          "match_percent": 78, "matched_genres": ["日常治愈"], "source": "llm"
        }
      }
    ],
    "meta": { "agent_chain": ["A1", "A3", "A4", "A5"], "degraded": [], "pool_size": 47 }
  }
}
```

---

#### `GET /recommend/{anime_id}/explain` —— 单条推荐解释

> 前端点击卡片上的「为什么推荐这个」时调用。同用户同番同风格命中 `recommend_explain` 唯一键时直接复用。

**Query**：`style=concise|detailed|casual`（默认 `concise`）

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 180,
  "data": {
    "anime_id": 1407,
    "title": "咒术回战",
    "reason": "你近期观看了《进击的巨人》《鬼灭之刃》等热血战斗番，本作同属该题材，题材匹配度 87%。",
    "core_items": [{ "title": "进击的巨人", "weight": 0.31 }],
    "match_percent": 87,
    "matched_genres": ["热血战斗"],
    "source": "llm",
    "prompt_ver": "explain_v1.0"
  }
}
```

| 错误码 | 场景 |
|---|---|
| `40401` | 该动漫不在当前用户的推荐结果中（无解释信号） |
| `60601` | LLM 超时 → 返回 `source=template` 而非报错 |

---

#### `POST /recommend/feedback`

**请求**

```jsonc
{
  "anime_id": 1407,
  "scene": 0,             // 同 recommend_result.scene
  "action": 1,            // 0=曝光 1=点击 2=收藏 3=不感兴趣 4=感兴趣
  "position": 1,          // 推荐位次（可选）
  "reason": "看过了",      // action=3 时可选
  "batch_id": "20260912_0300"
}
```

**响应**：`{"code": 0, "message": "ok", "data": {"recorded": true}}`

> 曝光埋点前端在卡片进入视口时调用（`action=0`），这是 CTR 分母的来源。

---

### 3.4 对话推荐模块

#### `POST /chat/session`

**响应**：`{"code":0,"data":{"session_id":"sess_9a8b7c","expires_in":1800}}`

#### `POST /chat/message` —— SSE 流式

**请求**

```jsonc
{ "session_id": "sess_9a8b7c", "message": "有没有类似咒术回战但是更轻松点的" }
```

**响应**：`Content-Type: text/event-stream`

```
data: {"type":"tool_call","tool":"search_anime","args":{"keyword":"咒术回战"}}

data: {"type":"tool_result","tool":"search_anime","count":1}

data: {"type":"chunk","content":"《咒术回战》是热血战斗路线，如果想轻松一点，可以试试这两部："}

data: {"type":"cards","cards":[{"anime_id":50265,"title":"间谍过家家","reason":"战斗元素配上家庭喜剧，节奏轻快"}]}

data: {"type":"done","session_id":"sess_9a8b7c","tool_calls":2,"tokens_used":412,"elapsed_ms":2380}
```

| 帧类型 | `type` | 说明 |
|---|---|---|
| 工具调用 | `tool_call` | 前端可显示"正在查询…" |
| 工具结果 | `tool_result` | — |
| 文本片段 | `chunk` | 逐段拼接显示 |
| 推荐卡片 | `cards` | 结构化推荐结果 |
| 结束 | `done` | 含统计信息 |
| 错误 | `error` | 含 `code`，前端展示友好文案 |

| 错误码 | 场景 |
|---|---|
| `60502` | 工具调用超过 3 次 → 强制收敛并继续输出 |
| `60503` | 上下文过长 → 截断历史后继续 |
| `60601` | LLM 超时 → `error` 帧 + 引导话术 |

---

### 3.5 分析模块

#### `GET /analysis/interest-radar`

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 35,
  "data": {
    "radar": [
      { "genre_id": 1, "genre": "热血战斗", "value": 0.82, "count": 46 },
      { "genre_id": 5, "genre": "悬疑推理", "value": 0.61, "count": 23 },
      { "genre_id": 3, "genre": "日常治愈", "value": 0.23, "count": 11 }
      // … 共 12 项，未涉及的 value=0
    ],
    "summary_text": "这是一位偏好热血战斗与悬疑推理的高活跃度追番用户。",
    "user_tag": "热血悬疑高活跃",
    "interest_capsules": [
      { "capsule_id": 0, "label": "热血战斗", "strength": 0.44 },
      { "capsule_id": 1, "label": "悬疑推理", "strength": 0.28 },
      { "capsule_id": 2, "label": "科幻机战", "strength": 0.19 },
      { "capsule_id": 3, "label": "日常治愈", "strength": 0.09 }
    ]
  }
}
```

#### `GET /analysis/drift-trend`

**Query**：`granularity=quarter`（默认）、`start=2024-01`、`end=2026-09`

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 60,
  "data": {
    "trend": [
      { "period": "2025Q1", "genres": { "热血战斗": 0.62, "悬疑推理": 0.21, "日常治愈": 0.17 } },
      { "period": "2025Q2", "genres": { "热血战斗": 0.31, "悬疑推理": 0.47, "日常治愈": 0.22 } }
    ]
  }
}
```

#### `GET /analysis/drift-points`

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 70,
  "data": {
    "drift_points": [
      { "period": "2025Q2", "js_divergence": 0.47, "from": "热血战斗", "to": "悬疑推理", "severity": "high" }
    ],
    "interpretation": "你的追番口味在 2025 年第二季度发生了明显转变，从热血战斗类逐渐转向悬疑推理类。",
    "threshold": 0.35
  }
}
```

#### `POST /analysis/agent-invoke` —— Agent 调试接口

> **仅开发/调试环境启用**（`APP_ENV != prod`）。用于答辩演示"直接观察某个 Agent 的行为"。

**请求**

```jsonc
{ "agent": "recall", "action": "recall.sequence", "payload": { "user_id": 1024, "top_k_per_interest": 10 } }
```

**响应**：透传该 Agent 的 `RecallOutput` + `meta.elapsed_ms`。

#### `GET /analysis/profile-summary`

兴趣分析页"概览区"的一次性数据源：A1 画像（与推荐同源）+ watch_record 聚合分布。

**响应**（节选）

```jsonc
{
  "data": {
    "profile": { "top_genres": [...], "activity_label": "high", "watch_intensity": 3.1,
                 "avg_rating_tendency": 7.8, "dropped_rate": 0.05, "total_records": 57,
                 "user_tag": "热血党", "summary_text": "..." },
    "status_dist": { "想看": 0, "在看": 2, "已看": 55, "弃番": 1 },
    "rating_dist": [ { "rating": 1, "count": 0 }, ... { "rating": 10, "count": 3 } ],
    "monthly_counts": [ { "period": "2025-10", "count": 6 }, ... ]   // 近 12 月
  }
}
```

#### `GET /animes/{anime_id}/reviews`

本站文字评价列表（匿名可浏览，与 `/similar` 同级公开）。按 `updated_at` 倒序，
只收录 `review` 非空的记录；昵称+头像级别匿名化，不暴露 user_id。

**响应**（节选）：`data.list[] = { nickname, avatar_url, rating, status, review, updated_at }`

**写入**：复用 `PUT /records/{id}` —— `RecordPatch` 新增 `review` 字段（≤500 字，
空串 = 清除），与 `rating`（1-10）并列、可只填其一。

#### `POST /users/me/avatar`

multipart 表单上传（字段名 `file`），支持 jpg/png/webp/gif，≤2MB。
落盘 `data/uploads/`，经 `/static/uploads/` 托管，返回相对路径并写入 `user.avatar_url`。

```jsonc
{ "code": 0, "message": "已上传", "data": { "avatar_url": "/static/uploads/avatar_1_tr_xxxx.png" } }
```

---

### 3.6 管理后台模块

#### `GET /admin/metrics`

**Query**：`date_from`、`date_to`、`scene`、`is_cold_start`

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 88,
  "data": {
    "online": {
      "exposure": 128400, "click": 10272, "fav": 1841,
      "ctr": 0.0800, "cvr": 0.1792, "cvrr": 0.1434
    },
    "offline": {
      "hr5": null, "hr10": null, "ndcg5": null, "ndcg10": null,
      "model_ver": "multi_interest_content_v1", "note": "离线指标由 scripts/run_experiments.py 回写"
    },
    "trend": [
      { "date": "2026-09-10", "ctr": 0.0781, "cvr": 0.1712 },
      { "date": "2026-09-11", "ctr": 0.0802, "cvr": 0.1798 }
    ],
    "anomalies": []
  }
}
```

> ⚠️ 离线指标 `null` 表示**实验尚未跑完**，不是 0。前端必须区分展示。

#### `GET /admin/cold-start`

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 45,
  "data": {
    "pool_size": 47,
    "funnel": { "exposure": 8600, "click": 774, "fav": 203 },
    "ctr": 0.0900, "fav_rate": 0.0236,
    "by_season": [
      { "season": "2026Q3", "count": 47, "exposure": 8600, "ctr": 0.0900 },
      { "season": "2026Q2", "count": 63, "exposure": 12400, "ctr": 0.0821 }
    ],
    "top_items": [
      { "anime_id": 20301, "title": "新番A", "exposure": 1204, "click": 132, "ctr": 0.1096 }
    ]
  }
}
```

#### `GET /admin/agents/health`

**响应**

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_x", "elapsed_ms": 12,
  "data": {
    "agents": [
      { "agent_id": "A0", "name": "orchestrator", "status": 1, "status_label": "正常",
        "health_score": 0.999, "avg_elapsed_ms": 12, "p95_elapsed_ms": 31,
        "success_cnt": 102400, "fail_cnt": 12, "timeout_cnt": 3, "degrade_cnt": 41 },
      { "agent_id": "A5", "name": "explain", "status": 2, "status_label": "降级中",
        "health_score": 0.94, "avg_elapsed_ms": 186, "p95_elapsed_ms": 220,
        "success_cnt": 98000, "fail_cnt": 210, "timeout_cnt": 190, "degrade_cnt": 1240 }
    ],
    "global": { "degrade_rate": 0.0128, "llm_timeout_rate": 0.0019, "circuit_open": [] }
  }
}
```

#### `GET /admin/traces/{trace_id}`

**响应**：返回该 `trace_id` 的完整调用树

```jsonc
{
  "code": 0, "message": "ok", "trace_id": "tr_7f3a9c21e0b4", "elapsed_ms": 20,
  "data": {
    "trace_id": "tr_7f3a9c21e0b4",
    "total_elapsed_ms": 265,
    "spans": [
      { "from_agent": "A0", "to_agent": "A1", "action": "profile.get", "status": 0, "elapsed_ms": 8, "tokens_used": 0 },
      { "from_agent": "A0", "to_agent": "A2", "action": "recall.sequence", "status": 0, "elapsed_ms": 35, "tokens_used": 0 },
      { "from_agent": "A0", "to_agent": "A3", "action": "content.retrieve", "status": 0, "elapsed_ms": 18, "tokens_used": 0 },
      { "from_agent": "A0", "to_agent": "A4", "action": "rank.fusion", "status": 0, "elapsed_ms": 14, "tokens_used": 0 },
      { "from_agent": "A0", "to_agent": "A5", "action": "explain.generate", "status": 1, "elapsed_ms": 186, "tokens_used": 412, "detail": {"degraded_reason": "llm_timeout"} }
    ]
  }
}
```

---

## 四、Agent 内部接口

### 4.1 `POST /internal/agents/{name}`

> **不是 RESTful 业务接口**，而是 Agent 层"进程内调用可切换为跨进程调用"的传输适配端点。仅内网可达。

**请求**

```jsonc
{
  "header": {
    "msg_id": "01H8X2K4M9P7Q1R3S5T7V9W2",
    "trace_id": "tr_7f3a9c21e0b4",
    "parent_msg_id": null,
    "from": "A0", "to": "A2",
    "type": "request", "action": "recall.sequence",
    "priority": "P0", "timestamp": 1757654321123,
    "timeout_ms": 200, "retry_count": 0, "schema_version": "1.0"
  },
  "context": { "user_id": 1024, "session_id": null, "locale": "zh-CN", "app_env": "prod" },
  "payload": { "user_seq": [1201, 5114, 34599], "top_k_per_interest": 50 }
}
```

**响应**：完整 Envelope（同结构 + `meta`）

**鉴权**：Header `X-Internal-Token: <INTERNAL_TOKEN>`，由 `.env` 配置；请求 IP 必须在白名单 `10.0.0.0/8, 172.16.0.0/12, 127.0.0.1`。

**Agent 名称与 action 对照**

| `name` | 支持的 `action` |
|---|---|
| `orchestrator` | `orchestrate.route` |
| `profile` | `profile.get`、`profile.refresh` |
| `recall` | `recall.sequence`、`recall.batch` |
| `coldstart` | `content.retrieve` |
| `fusion` | `rank.fusion` |
| `explain` | `explain.generate` |
| `drift` | `drift.analyze` |
| `chat` | `chat.reply` |
| `dataops` | `dataops.sync_new_anime`、`dataops.rebuild_index` |
| `eval` | `eval.offline_metrics` |

---

## 五、前端调用示例

### 5.1 axios 实例与拦截器

```ts
// src/api/request.ts
import axios from 'axios'

const request = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 15000,
})

request.interceptors.request.use((cfg) => {
  const token = localStorage.getItem('access_token')
  if (token) cfg.headers.Authorization = `Bearer ${token}`
  return cfg
})

request.interceptors.response.use(
  (res) => {
    const { code, message, data } = res.data
    if (code !== 0) {
      ElMessage.error(message)
      return Promise.reject(new Error(message))
    }
    return data                     // 直接返回业务载荷
  },
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('access_token')
      router.push('/login')
    }
    return Promise.reject(err)
  },
)

export default request
```

### 5.2 推荐列表

```ts
// src/api/recommend.ts
import request from './request'

export const getFeed = (size = 20, withExplain = true) =>
  request.get('/recommend/feed', { params: { size, with_explain: withExplain } })

export const getNewAnime = (season?: string) =>
  request.get('/recommend/new-anime', { params: { season } })

export const explainOne = (animeId: number, style = 'concise') =>
  request.get(`/recommend/${animeId}/explain`, { params: { style } })

export const sendFeedback = (payload: FeedbackPayload) =>
  request.post('/recommend/feedback', payload)
```

### 5.3 页面中消费（含曝光埋点）

```vue
<!-- src/views/RecommendFeed.vue 片段 -->
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getFeed, sendFeedback } from '@/api/recommend'

const items = ref<RecItem[]>([])
const meta = ref<RecMeta | null>(null)
const observer = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (e.isIntersecting) {
      const id = Number((e.target as HTMLElement).dataset.animeId)
      sendFeedback({ anime_id: id, scene: 0, action: 0, position: 0, batch_id: meta.value?.batch_id })
      observer.unobserve(e.target)
    }
  })
}, { threshold: 0.5 })

onMounted(async () => {
  const data = await getFeed(20, true)
  items.value = data.items
  meta.value = data.meta
  if (data.meta.degraded.length) {
    ElMessage.warning('部分推荐能力暂时降级，结果可能略有差异')
  }
})
</script>

<template>
  <AnimeCard v-for="it in items" :key="it.anime.id" :item="it" @click="onClick(it)" />
  <ExplainBadge v-if="items[0]?.explain" :explain="items[0].explain" />
</template>
```

### 5.4 对话流式（SSE）

```ts
// src/api/chat.ts
export function chatStream(sessionId: string, message: string, onFrame: (f: ChatFrame) => void) {
  const token = localStorage.getItem('access_token')
  return fetch(`${import.meta.env.VITE_API_BASE_URL}/chat/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ session_id: sessionId, message }),
  }).then(async (resp) => {
    const reader = resp.body!.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) onFrame(JSON.parse(line.slice(6)))
      }
    }
  })
}
```

---

## 六、HTTP 状态码与业务码映射

| HTTP | 业务码 | 含义 | 前端处理 |
|---|---|---|---|
| 200 | 0 | 成功 | 正常渲染 |
| 200 | `60401` | 已降级但可用 | 正常渲染 + 顶部提示 |
| 400 | `40001` | 参数错误 | 表单标红 |
| 401 | `40101` | 未登录/失效 | 跳登录页 |
| 403 | `40301` | 无权限 | 提示无权限 |
| 404 | `40401` | 资源不存在 | 空状态页 |
| 409 | `40901` | 冲突/重复 | 提示已存在 |
| 429 | `42901` | 限流 | 按 `Retry-After` 重试 |
| 500 | `50001` | 服务端异常 | 提示稍后重试 + 上报 `trace_id` |
| 503 | `50202` | 依赖不可用 | 提示服务维护中 |

> **重要**：降级（`60401`）与部分 Agent 失败**一律返回 HTTP 200**，通过 `meta.degraded` 表达。这样前端不会把"新番候选没取到"当成错误弹窗。

---

## 七、限流与超时

| 接口 | 限流 | 服务端超时 |
|---|---|---|
| `/auth/login` | 10 次/分钟/IP | 5s |
| `/recommend/feed` | 60 次/分钟/用户 | 1s |
| `/recommend/*/explain` | 30 次/分钟/用户 | 1s |
| `/chat/message` | 20 次/分钟/用户 | 15s（SSE） |
| `/analysis/*` | 30 次/分钟/用户 | 3s |
| `/admin/*` | 120 次/分钟/用户 | 10s |
| `/internal/agents/*` | 不限（内网） | 按 Envelope `timeout_ms` |

超时响应：HTTP 504 + `{"code": 50301, "message": "处理超时，请稍后重试"}`。

---

## 八、接口开发检查清单

新增一个接口时必须逐项确认：

- [ ] 已在本文第二节总览表登记
- [ ] 请求/响应使用 `server/schemas/` 中的 Pydantic 模型，未直接用 `dict`
- [ ] 响应包裹统一响应体（由 `core/response.py` 自动完成，不要手写）
- [ ] 错误码使用本文与协议文档中已定义的值，未新增未登记的错误码
- [ ] 鉴权依赖已在路由声明（`Depends(get_current_user)` / `Depends(require_admin)`）
- [ ] 有对应的接口测试（`tests/test_api/`）
- [ ] 涉及 Agent 的接口已确认降级行为，并通过 `meta.degraded` 透出
- [ ] 已在 `frontend/src/api/` 中补充对应的请求封装

---

## 九、相关文档

- 数据表 → [database-design.md](database-design.md)
- 错误码与降级 → [agent-interaction-protocol.md](agent-interaction-protocol.md)
- Agent 输出结构 → [agent-prompt-design.md](agent-prompt-design.md)
- 指标口径 → [evaluation-plan.md](evaluation-plan.md)

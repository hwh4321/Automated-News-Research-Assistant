# 自动新闻调研助手

基于 LangGraph + LangChain 的 AI 新闻调研自动化系统，支持**自动搜索 → 总结 → 生成报告 → 发送邮件**全流程。

## 架构

```
用户输入 → Plan（规划子任务）→ Search（多源搜索）→ Summarize（AI摘要）
         → Report（生成报告）→ Email（邮件发送）
```

### 四种运行模式

| 模式 | 命令 | 特点 |
|------|------|------|
| **LangGraph 工作流** | `python main.py run "主题"` | 确定性 DAG 管道，Plan → Search → Summarize → Report → Email |
| **ReAct Agent** | `python main.py react "主题"` | LLM 自主推理 + 工具调用循环（Think → Act → Observe） |
| **Plan-and-Execute** | `python main.py plan "主题"` | 先规划子任务清单，再逐步执行，过程可追溯 |
| **RAG 问答** | `python main.py ask "问题"` | 基于历史调研资料的检索增强问答 |

## 项目结构

```
自动新闻调研助手/
├── .env.example              # API 密钥配置模板
├── requirements.txt          # Python 依赖
├── config.py                 # 配置管理（dataclass + 环境变量）
├── main.py                   # CLI 入口
│
├── utils/
│   ├── __init__.py           # parse_json_response() 通用工具
│   └── llm.py                # ChatOpenAI + ReAct 工具调用循环
│
├── tools/                    # Function Calling 工具集（@tool 装饰器）
│   ├── search_tool.py        # 新闻搜索（SerpAPI / NewsAPI / Tavily）
│   ├── summarize_tool.py     # LLM 摘要（ChatPromptTemplate）
│   ├── report_tool.py        # Markdown 研究报告
│   ├── email_tool.py         # SMTP 邮件发送
│   ├── qa_chain.py           # RAG 检索增强问答链（create_stuff_documents_chain）
│   └── registry.py           # 工具注册与分发
│
├── graph/                    # LangGraph 工作流
│   ├── state.py              # ResearchState 定义（TypedDict + Annotated reducer）
│   ├── nodes.py              # 5 个图节点
│   └── workflow.py           # StateGraph 编译器 + 条件路由
│
├── agents/                   # 智能体
│   ├── react_agent.py        # ReAct 模式（LangChain Messages）
│   └── planner.py            # Plan-and-Execute 模式
│
└── memory/                   # 混合记忆系统
    ├── vector_store.py       # ChromaDB 向量存储（语义检索）
    ├── relational_store.py   # SQLite 关系存储（3 张表）
    └── session.py            # 会话管理（Redis + 内存降级）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
```

编辑 `.env`，填入以下密钥：

| 变量 | 说明 | 获取地址 |
|------|------|---------|
| `LLM_API_KEY` | 大模型 API 密钥 | OpenAI / DeepSeek / 其他兼容厂商 |
| `SEARCH_API_KEY` | 搜索引擎 API 密钥 | SerpAPI / NewsAPI / Tavily |
| `SMTP_USER` / `SMTP_PASSWORD` | 邮件发送账号 | Gmail 等 SMTP 服务 |

**大模型兼容性**：支持所有 OpenAI API 兼容的模型服务（DeepSeek、通义千问、GLM 等），只需修改 `LLM_BASE_URL` 和 `LLM_MODEL`。

**搜索引擎选择**：`SEARCH_ENGINE` 设为 `serpapi`（推荐）、`newsapi` 或 `tavily`。

### 3. 运行

```bash
# LangGraph 工作流
python main.py run "2026年AI行业最新动态"

# ReAct 智能体
python main.py react "OpenAI新产品发布"

# Plan-and-Execute（附带邮件发送）
python main.py plan "量子计算技术突破" --email you@example.com

# RAG 问答 — 基于历史调研回答
python main.py ask "OpenAI最近发布了什么产品"

# 查看存储统计
python main.py stats
```

## 技术要点

### LangChain 集成策略

**用了 LangChain 的：**

| 模块 | LangChain 组件 | 替代了 |
|------|---------------|--------|
| 所有工具 | `@tool` 装饰器 | 手写 JSON Schema |
| `qa_chain.py` | LCEL `retriever \| format \| prompt \| LLM \| StrOutput` | `create_stuff_documents_chain` 旧版写法 |
| `utils/llm.py` | `ChatOpenAI` + `.bind_tools()` | 裸 `openai.OpenAI` |
| Prompt 管理 | `ChatPromptTemplate.from_messages()` | 裸字符串拼接 |

**没用 LangChain 的（保持简洁）：**

| 模块 | 保持原生 | 原因 |
|------|---------|------|
| ChromaDB | 直连 `chromadb` | 避免 embedding 兼容问题，LangChain wrapper 无实质增益 |
| SMTP | 标准库 `smtplib` | 单次发送，不需要抽象 |
| HTTP 搜索 | `requests` | 几个 API 调用，不需要 DocumentLoader |
| LangGraph 工作流 | 不变 | 本身就是 LangChain 生态 |

### 混合记忆架构

```
┌─────────────────────────────────────────┐
│                记忆系统                    │
├──────────────┬──────────────┬─────────────┤
│  ChromaDB    │   SQLite     │   Redis     │
│  (向量存储)   │  (关系存储)  │  (会话缓存)  │
├──────────────┼──────────────┼─────────────┤
│ 语义检索      │ 结构化查询    │ 分布式状态   │
│ 相似度匹配    │ 历史记录      │ 自动过期     │
│ 历史上下文    │ 报告持久化    │ 可降级内存   │
└──────────────┴──────────────┴─────────────┘
```

- **ChromaDB**：存储搜索结果嵌入向量，支持语义相似度检索
- **SQLite**：`sessions`、`queries`、`reports` 三表存储结构化数据
- **Redis**：会话状态缓存（TTL 自动过期），未配置时自动降级为内存字典

### ReAct 模式

```
用户输入 → System Prompt → LLM 推理 → 调用工具 → 观察结果 → 继续推理 → 最终回答
              ↑                                        ↓
              └──────────── 循环（max 5 轮）──────────────┘
```

### Plan-and-Execute 模式

```
用户输入 → Planner（拆解为 3-5 个子任务）→ 逐个执行 → 汇总 → 生成报告
```

### Function Calling 工具

| 工具名 | 定义方式 | 功能 |
|--------|---------|------|
| `search_news` | `@tool` | 多引擎新闻搜索 |
| `summarize_text` | `@tool` + `ChatPromptTemplate` | LLM 文本摘要 |
| `generate_report` | `@tool` + `ChatPromptTemplate` | Markdown 研究报告 |
| `send_email` | `@tool` | SMTP 邮件发送 |
| `ask_research` | `@tool` + LCEL 管道 | RAG 问答（检索 → 格式化 → Prompt → LLM → 输出） |

### 安全性

- 所有 API 密钥通过 `.env` 环境变量管理，不写入代码
- SQLite 使用参数化查询，杜绝 SQL 注入
- SMTP 强制 STARTTLS 加密
- `.env` 已加入 `.gitignore`

## 依赖

```
langgraph>=0.2.0            # 有状态工作流编排
langchain-openai>=0.2.0     # ChatOpenAI (替代原生 openai SDK)
chromadb>=0.5.0              # 向量数据库
python-dotenv>=1.0.0         # 环境变量管理
requests>=2.31.0             # HTTP 请求（搜索 API）
redis>=5.0.0                 # 可选，会话持久化
```

## Python 语法糖使用

项目大量使用 Python 3.10+ 现代语法：

| 语法 | 使用位置 |
|------|---------|
| `match/case` | `search_tool.py` 搜索引擎分发、`main.py` CLI 路由 |
| `:=` 海象运算符 | `registry.py` 工具查找、`nodes.py` 结果判空、`relational_store.py` 查询取值、`session.py` Redis 加载 |
| `dataclass` + `asdict` | `config.py`、`session.py` 序列化 |
| `pathlib.Path` | `relational_store.py`、`vector_store.py` 路径操作 |
| `cached_property` | `config.py` Redis 检测 |
| `Annotated[T, add]` reducer | `state.py` 列表累积归并 |
| `list/dict` 推导式 | 全局搜索 API 响应解析 |
| `|` 联合类型 | `Session | None`、`str | list | dict` |
| `ChatPromptTemplate` | `summarize_tool.py`、`report_tool.py`、`qa_chain.py`、`nodes.py`、`planner.py` |
| LCEL `\|` 管道组合 | `qa_chain.py` RAG 链：`retriever \| format \| prompt \| LLM \| output` |

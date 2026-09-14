# 远程 Worker 部署

Mac 负责 ChromaDB、BM25、LLM 和 UI，只把 embedding 与 reranker 两类模型推理交给
Windows 显卡机器；两端通过 Tailscale 直连，不经公网。

## Mac 本地 CPU 模式

先装本地模型依赖：

```bash
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --inexact --extra local-models
```

`project/.env`：

```env
RAG_QA_COMPUTE_BACKEND=local
RAG_QA_DEVICE=cpu
RAG_QA_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RAG_QA_RERANKER_MODEL=BAAI/bge-reranker-base
```

## Windows Worker（4070 Super）

拉取同一分支后，在 PowerShell 中创建环境并安装依赖：

```powershell
conda env create -f environment.yml
conda activate rag-textbook-qa
$env:UV_PROJECT_ENVIRONMENT=$env:CONDA_PREFIX
uv sync --inexact --extra worker --extra local-models
```

用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成随机 token 写入 Windows 的
`project/.env`。**Worker token 必须是非空 ASCII 字符串**，不能含首尾空格、内部空白或控制字符：

```env
RAG_QA_WORKER_TOKEN=替换为随机token
RAG_QA_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RAG_QA_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_QA_DEVICE=cuda
```

用 `tailscale ip -4` 查台式机的 Tailscale IP，只监听该地址：

```powershell
rag-qa worker serve --host 100.x.y.z --port 8765 --device cuda
```

**不要把 Worker 端口映射到公网。** 监听非 localhost 地址时程序会强制要求
`RAG_QA_WORKER_TOKEN`。若终端进程里的 token 与 `project/.env` 不同，以进程环境变量为准，
命令会给出不含 token 内容的警告；改完 token 要重启 Worker。

### 一键启动脚本

配置完成后可从仓库根目录直接运行，不需要先 `conda activate`：

```powershell
.\scripts\windows\start-worker.ps1
```

脚本按自身位置定位仓库，自动查找 Conda 和 Tailscale IPv4，检查 `project/.env` 里的 token 格式，
确认 8765 端口空闲后用 `conda run` 启动 CUDA Worker。它不显示 token，不改防火墙、开机启动项或
持久环境变量；即使当前 PowerShell 残留旧的 Worker 配置，也只为本次子进程清除这些覆盖值，
一律以仓库的 `project/.env` 为准。

也可用绝对路径从别处启动，或覆盖环境名和端口：

```powershell
& "D:\CodeField\rag-textbook-qa-worker\scripts\windows\start-worker.ps1"
.\scripts\windows\start-worker.ps1 -EnvironmentName rag-textbook-qa -Port 8765
```

要把首次请求的模型加载等待挪到启动阶段，显式启用预热：

```powershell
.\scripts\windows\start-worker.ps1 -Warmup
# 等价 CLI 参数：rag-qa worker serve ... --warmup
```

预热只在 Windows 本地各跑一次最小的 embedding 和 rerank 推理，不调用 LLM 或外部 API，
也不改教材索引。启用后 Worker 会等两个模型都加载完再开始监听；不加 `-Warmup` 时保持原有的
首次请求懒加载行为。

## Mac 连接远程 Worker

Mac 的 `project/.env` 写入相同 token 和 Windows 的 Tailscale 地址：

```env
RAG_QA_COMPUTE_BACKEND=remote
RAG_QA_REMOTE_URL=http://100.x.y.z:8765
RAG_QA_WORKER_TOKEN=与Windows相同的随机token
RAG_QA_REMOTE_TIMEOUT=120
RAG_QA_QUERY_FALLBACK_TO_LOCAL=false
RAG_QA_EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
RAG_QA_RERANKER_MODEL=BAAI/bge-reranker-base
```

先做健康检查：

```bash
rag-qa worker check
rag-qa worker check --json   # 结构化输出
```

它只请求 `/health`，校验认证、协议版本、设备以及 embedding/reranker 的模型指纹，
不调用推理接口，也不输出 token。`rag-qa doctor` 只检查当前选择的后端配置，不连接 Worker。
Worker 在首次收到 embedding 或 rerank 请求时才加载并下载模型。

要启用查询回退，Mac 还需安装 `local-models` 并把 `RAG_QA_QUERY_FALLBACK_TO_LOCAL` 改为 `true`。
回退只针对**查询**且只在瞬时故障（如连接超时）时发生；认证失败和模型指纹不一致始终直接报错，
索引构建则从不回退——一次构建必须全程使用同一个后端，否则向量会不同源。

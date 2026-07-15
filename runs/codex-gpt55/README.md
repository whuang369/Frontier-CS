# FrontierCS · Codex + GPT-5.5 批量评测

`runs/codex-gpt55/run.py`:可断点续跑的并行评测脚本,用 `codex` agent + `openai/gpt-5.5`
跑 `algorithmic` track 全部题目,结果写入 `runs/codex-gpt55/results.jsonl`。
所需的代码改动已合入本分支,**clone 下来即可用,无需打补丁**。

---

## 1. 获取代码

```bash
git clone -b codex-gpt55-sweep https://github.com/whuang369/Frontier-CS.git FrontierCS
cd FrontierCS
uv sync                     # 建 venv(需 uv ≥ 0.9,Python 3.12)
```

## 2. 安装 harbor

`frontier harbor trial` 内部会调用外部 `harbor` CLI,装一下即可:

```bash
uv tool install harbor      # v0.18.0,装到 ~/.local/bin
```

## 3. Docker

确保 **Docker Desktop(macOS)/ docker daemon(Linux)已启动** —— 每个 trial 在容器里跑。

## 4. OpenAI Key

在 repo 根目录 `.env` 里放(已 gitignore):

```
OPENAI_API_KEY=sk-你的key
```

## 5. 运行

### 5a. 冒烟测试(单题,验证环境;留空 key 不花 token)

agent 会在首次调用 OpenAI 时失败退出,只验证 Docker / task 生成 / harbor 链路是否通:

```bash
OPENAI_API_KEY="" uv run --no-sync frontier harbor trial algorithmic 0 \
  -a codex -m openai/gpt-5.5 --uv \
  --agent-kwarg reasoning_effort=high --agent-timeout 120 --json
```

看到最后输出一段带 `"trial_status": "scored"` 的 JSON 即环境 OK。

### 5b. 完整评测(全部题目,后台)

```bash
nohup python3 runs/codex-gpt55/run.py > runs/codex-gpt55/run.out 2>&1 &
tail -f runs/codex-gpt55/run.out
```

---

## 6. 输出 & 断点续跑

写在 `runs/codex-gpt55/`:
- `results.jsonl` —— 每题一行结果(reward / score / tokens / trial_dir / …)
- `logs/algorithmic_<id>.log` —— 每题 Harbor stdout
- `heartbeat.txt` / `run.pid` —— 运行状态
- Harbor trial 目录在 `<repo>/.frontier-cs/harbor/trials/`

**断点续跑**:重启脚本时,已 `scored`(reward 非空且 status=scored)的题自动跳过,
失败的会重试。再跑一次 `nohup python3 ... &` 即可接着来。

## 7. 常用参数(`run.py` 顶部 config 区)

| 变量 | 默认 | 含义 |
|---|---|---|
| `MODEL` | `openai/gpt-5.5` | 模型 |
| `AGENT` | `codex` | agent |
| `TRACK` | `algorithmic` | 赛道(也可 `2.0`) |
| `CONCURRENCY` | `3` | 同时跑几道题;高了更快但更吃 API/内存 |
| `AGENT_TIMEOUT` | `18000`(5h) | 每题 agent 执行上限 |
| `AGENT_KWARGS` | `reasoning_effort=high, reasoning_summary=detailed` | 转发给 codex 的调参 |
| `PROBLEMS` | `None`(全部) | 只跑某几题就填 id 列表,如 `["0","257"]` |

---

## 附:一页速通

```bash
git clone -b codex-gpt55-sweep https://github.com/whuang369/Frontier-CS.git FrontierCS && cd FrontierCS
uv sync
uv tool install harbor
echo 'OPENAI_API_KEY=sk-...' >> .env
# 冒烟测试(不花钱):
OPENAI_API_KEY="" uv run --no-sync frontier harbor trial algorithmic 0 -a codex -m openai/gpt-5.5 --uv --agent-kwarg reasoning_effort=high --agent-timeout 120 --json
# 完整跑:
nohup python3 runs/codex-gpt55/run.py > runs/codex-gpt55/run.out 2>&1 &
tail -f runs/codex-gpt55/run.out
```

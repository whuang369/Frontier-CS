# FrontierCS · Codex + GPT-5.5 Batch Evaluation

`runs/codex-gpt55/run.py`: a resumable parallel evaluation script that runs the `codex`
agent + `openai/gpt-5.5` over the whole `algorithmic` track via `frontier harbor trial`,
writing results to `runs/codex-gpt55/results.jsonl`. All required code changes are already
merged into this branch — **just clone and run, no patch needed.**

---

## 1. Get the code

```bash
git clone -b codex-gpt55-sweep https://github.com/whuang369/Frontier-CS.git FrontierCS
cd FrontierCS
uv sync                     # build the venv (needs uv >= 0.9, Python 3.12)
```

## 2. Install harbor

`frontier harbor trial` calls the external `harbor` CLI under the hood, so install it:

```bash
uv tool install harbor      # v0.18.0, installed to ~/.local/bin
```

## 3. Docker

Make sure **Docker Desktop (macOS) / the docker daemon (Linux) is running** — each trial
runs inside a container.

## 4. OpenAI key

Put your key in `.env` at the repo root (already gitignored):

```
OPENAI_API_KEY=sk-your-key
# Optional: a custom / proxy endpoint. If set, it is forwarded to the agent
# automatically (via --agent-env). Leave it unset to use the default OpenAI API.
OPENAI_BASE_URL=https://your-endpoint/v1
```

## 5. Run

### 5a. Smoke test (single problem, verify the environment; empty key = no tokens spent)

The agent fails fast at its first OpenAI call, so this only verifies the
Docker / task-generation / harbor pipeline:

```bash
OPENAI_API_KEY="" uv run --no-sync frontier harbor trial algorithmic 0 \
  -a codex -m openai/gpt-5.5 --uv \
  --agent-kwarg reasoning_effort=high --agent-timeout 120 --json
```

A final JSON blob containing `"trial_status": "scored"` means the environment is OK.

### 5b. Full evaluation (all problems, in the background)

```bash
nohup python3 runs/codex-gpt55/run.py > runs/codex-gpt55/run.out 2>&1 &
tail -f runs/codex-gpt55/run.out
```

---

## 6. Outputs & resuming

Written under `runs/codex-gpt55/`:
- `results.jsonl` — one line per problem (reward / score / tokens / trial_dir / …)
- `logs/algorithmic_<id>.log` — per-problem Harbor stdout
- `heartbeat.txt` / `run.pid` — run state
- Harbor trial dirs live under `<repo>/.frontier-cs/harbor/trials/`

**Resuming**: on restart, problems already `scored` (reward present and status=scored) are
skipped; failed ones are retried. Just launch `nohup python3 ... &` again to continue.

## 7. Common settings (config block at the top of `run.py`)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `openai/gpt-5.5` | model |
| `AGENT` | `codex` | agent |
| `TRACK` | `algorithmic` | track (can also be `2.0`) |
| `CONCURRENCY` | `3` | trials in parallel; higher = faster but more API/memory load |
| `AGENT_TIMEOUT` | `18000` (5h) | per-problem agent execution ceiling |
| `AGENT_KWARGS` | `reasoning_effort=high, reasoning_summary=detailed` | kwargs forwarded to codex |
| `PROBLEMS` | `None` (all) | set an id list to run only some, e.g. `["0","257"]` |

---

## Appendix: one-page quickstart

```bash
git clone -b codex-gpt55-sweep https://github.com/whuang369/Frontier-CS.git FrontierCS && cd FrontierCS
uv sync
uv tool install harbor
echo 'OPENAI_API_KEY=sk-...' >> .env
# Smoke test (no tokens spent):
OPENAI_API_KEY="" uv run --no-sync frontier harbor trial algorithmic 0 -a codex -m openai/gpt-5.5 --uv --agent-kwarg reasoning_effort=high --agent-timeout 120 --json
# Full run:
nohup python3 runs/codex-gpt55/run.py > runs/codex-gpt55/run.out 2>&1 &
tail -f runs/codex-gpt55/run.out
```

#!/usr/bin/env python3
"""Resumable parallel agent-eval sweep: Codex + GPT-5.5 over a FrontierCS track,
through `frontier harbor trial`.

Configuration (per user request):
- Agent:   codex
- Model:   openai/gpt-5.5
- Configs: everything cranked to *high* -> reasoning_effort=high,
           reasoning_summary=detailed  (forwarded as Harbor --agent-kwarg).
- Timeout: 5h agent-execution ceiling per problem (AGENT_TIMEOUT = 18000s).
- Auth:    OPENAI_API_KEY and OPENAI_BASE_URL are intentionally left EMPTY.
           They are forwarded to the agent as empty strings via --agent-env,
           which *overrides* any ambient host env so the run stays credential-less
           until you fill them in below. With empty creds the codex CLI will fail
           fast at the first OpenAI call (no real tokens are spent); fill in the
           two constants below to do a real run.

Design goals (mirrors runs/gemini31pro/sweep.py):
- Parallel but bounded: at most CONCURRENCY trials at once (shared 224-core box).
- Keep traces: harbor trial dirs are kept (no --delete); per-trial stdout logged.
- Crash-tolerant + resumable: all state on disk; problems that produced a real
  score are skipped on restart, so re-launching continues where it left off.

Run:
  nohup python3 runs/codex-gpt55/run.py > runs/codex-gpt55/run.out 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

def _find_repo_root() -> Path:
    """Locate the FrontierCS repo root by walking up from this file until the
    dir containing `algorithmic/` and `src/frontier_cs/` is found — keeps the
    script portable (no hard-coded absolute path to edit per machine)."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "algorithmic").is_dir() and (parent / "src" / "frontier_cs").is_dir():
            return parent
    return here.parents[2]  # fallback for repo/runs/codex-gpt55/run.py


# ----------------------------- configuration ------------------------------
REPO = _find_repo_root()
RUN_DIR = Path(__file__).resolve().parent   # write outputs next to this script
LOG_DIR = RUN_DIR / "logs"
RESULTS = RUN_DIR / "results.jsonl"
HEARTBEAT = RUN_DIR / "heartbeat.txt"

AGENT = "codex"
MODEL = "openai/gpt-5.5"
TRACK = "algorithmic"                    # 2.0 (agentic) or algorithmic

# codex config knobs forwarded as Harbor --agent-kwarg. reasoning_effort is the
# main cost/quality lever (high = max reasoning = most tokens = most $).
# reasoning_summary only affects trace verbosity; drop it to shave a little cost.
AGENT_KWARGS = [
    "reasoning_effort=high",
    "reasoning_summary=detailed",
]

# --- Auth: leave BOTH empty to read OPENAI_API_KEY / OPENAI_BASE_URL from .env
#     (recommended: .env is gitignored). A non-empty value here overrides .env. ---
OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""

# Per-problem agent-execution ceiling: 5h, per requirement.
AGENT_TIMEOUT = 5 * 3600        # 5h (18000s)
VERIFIER_TIMEOUT: float | None = None  # None -> task default
# Concurrency = simultaneous trials. Validated 3-way with NO OpenAI rate limits.
# Higher = faster but more API load; if you see 429s, lower it.
CONCURRENCY = 3

# Explicit problem list, or None to auto-discover every problem in the track.
PROBLEMS: list[str] | None = None

# Trial wall-clock guard: agent ceiling + docker build + verifier/judge overhead.
# Kill a trial that wildly overruns so a hung trial can't wedge a worker forever.
BUILD_JUDGE_OVERHEAD = 90 * 60  # 1.5h for image build + judge
TRIAL_HARD_TIMEOUT = AGENT_TIMEOUT + BUILD_JUDGE_OVERHEAD


# ------------------------------- helpers ----------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        HEARTBEAT.write_text(line + "\n")
    except OSError:
        pass


def all_problem_ids() -> list[str]:
    if PROBLEMS is not None:
        return list(PROBLEMS)
    pdir = REPO / TRACK / "problems"
    ids = [p.name for p in pdir.iterdir() if p.is_dir()]
    # algorithmic ids are numeric; sort those numerically, others lexically.
    if all(i.isdigit() for i in ids):
        return sorted(ids, key=int)
    return sorted(ids)


def load_results() -> dict[str, dict]:
    """problem_id -> last result record."""
    out: dict[str, dict] = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[str(rec.get("problem_id"))] = rec
    return out


def is_done(rec: dict) -> bool:
    """Done == produced a real score (reward present and status 'scored').
    Infra/driver errors (no reward) are retried on the next pass."""
    return rec.get("reward") is not None and rec.get("trial_status") == "scored"


def extract_last_json(text: str) -> dict | None:
    """Return the last balanced top-level {...} object in text (the --json summary)."""
    import re

    starts = [m.start() for m in re.finditer(r"^\{", text, re.MULTILINE)]
    for start in reversed(starts):
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def read_dotenv() -> dict[str, str]:
    """Return KEY=VALUE pairs from REPO/.env (gitignored secret store)."""
    vals: dict[str, str] = {}
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    return vals


def run_trial(pid: str, env: dict) -> dict:
    """Run one harbor trial for problem `pid`; return a normalized result record."""
    log_path = LOG_DIR / f"{TRACK.replace('.', '_')}_{pid}.log"
    # OPENAI_API_KEY reaches codex via `env` (os.environ of the harbor process,
    # read by the agent) — deliberately not via --agent-env, so the secret key
    # never appears on the process command line. OPENAI_BASE_URL is not secret,
    # so we forward it explicitly to the agent via --agent-env to guarantee it
    # reaches the containerized agent regardless of harbor's host-env passthrough.
    cmd = [
        "uv", "run", "--no-sync", "frontier", "harbor", "trial", TRACK, pid,
        "-a", AGENT, "-m", MODEL, "--uv",
        "--agent-timeout", str(AGENT_TIMEOUT),
        "--json",
    ]
    for kw in AGENT_KWARGS:
        cmd += ["--agent-kwarg", kw]
    base_url = env.get("OPENAI_BASE_URL")
    if base_url:
        cmd += ["--agent-env", f"OPENAI_BASE_URL={base_url}"]
    wire_api = env.get("CODEX_WIRE_API")
    if wire_api:
        cmd += ["--agent-env", f"CODEX_WIRE_API={wire_api}"]
    if VERIFIER_TIMEOUT is not None:
        cmd += ["--verifier-timeout", str(VERIFIER_TIMEOUT)]

    started = datetime.now()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=TRIAL_HARD_TIMEOUT,
        )
        out = proc.stdout or ""
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out += "\n[run] TRIAL_HARD_TIMEOUT killed the trial\n"
        rc = -9
    log_path.write_text(out)

    payload = extract_last_json(out) or {}
    rec = {
        "problem_id": pid,
        "track": TRACK,
        "agent": AGENT,
        "model": MODEL,
        "reward": payload.get("reward"),
        "score": payload.get("score"),
        "score_unbounded": payload.get("score_unbounded"),
        "trial_status": payload.get("trial_status"),
        "agent_status": payload.get("agent_status"),
        "n_input_tokens": payload.get("n_input_tokens") or 0,
        "n_output_tokens": payload.get("n_output_tokens") or 0,
        "n_cache_tokens": payload.get("n_cache_tokens") or 0,
        "cost_usd": payload.get("cost_usd"),
        "successful_submissions": payload.get("successful_submissions"),
        "trial_dir": payload.get("trial_dir"),
        "error_message": payload.get("error_message"),
        "return_code": rc,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().isoformat(),
    }
    return rec


# -------------------------------- main ------------------------------------
def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "run.pid").write_text(str(os.getpid()) + "\n")

    # Resolve creds: constant (if set) > .env > ambient environment.
    env = os.environ.copy()
    dotvals = read_dotenv()
    api_key = OPENAI_API_KEY or dotvals.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY", "")
    base_url = OPENAI_BASE_URL or dotvals.get("OPENAI_BASE_URL") or env.get("OPENAI_BASE_URL", "")
    # The key reaches codex through env (os.environ of the harbor subprocess).
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    else:
        env.pop("OPENAI_BASE_URL", None)  # empty -> use the default OpenAI endpoint
    # Optional HTTP wire protocol for a custom node (responses|chat). Only used
    # with a custom OPENAI_BASE_URL + the force-HTTP patch (patch_codex_http.py);
    # forwarded to the codex agent via --agent-env in run_trial.
    wire_api = dotvals.get("CODEX_WIRE_API") or env.get("CODEX_WIRE_API", "")
    if wire_api:
        env["CODEX_WIRE_API"] = wire_api

    if not api_key:
        log("FATAL: no OPENAI_API_KEY found. Put it in .env (OPENAI_API_KEY=sk-...) "
            "or set the OPENAI_API_KEY constant. Aborting before spending nothing.")
        sys.exit(1)
    log(f"auth: OPENAI_API_KEY set (...{api_key[-4:]}), "
        f"base_url={base_url or 'default OpenAI endpoint'}")

    results = load_results()
    all_ids = all_problem_ids()
    todo = [pid for pid in all_ids if not is_done(results.get(pid, {}))]
    log(f"sweep start: {len(todo)} problems to run "
        f"({len(all_ids)} total, {len(all_ids) - len(todo)} already scored). "
        f"track={TRACK} agent={AGENT} model={MODEL} "
        f"concurrency={CONCURRENCY} agent_timeout={AGENT_TIMEOUT}s "
        f"kwargs={AGENT_KWARGS}")

    inflight: dict = {}  # future -> pid

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        while todo or inflight:
            # ---- reap finished ----
            for fut in [f for f in list(inflight) if f.done()]:
                pid = inflight.pop(fut)
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {
                        "problem_id": pid, "track": TRACK, "reward": None,
                        "trial_status": "driver_error", "error": repr(e),
                        "finished_at": datetime.now().isoformat(),
                    }
                append_jsonl(RESULTS, rec)
                results[pid] = rec
                note = ""
                if not is_done(rec):
                    note = " (will retry next pass)"
                log(f"done {pid}: reward={rec.get('reward')} "
                    f"status={rec.get('trial_status')} rc={rec.get('return_code')} "
                    f"cost={rec.get('cost_usd')} in_tok={rec.get('n_input_tokens')}"
                    f"{note}")

            # ---- dispatch ----
            while todo and len(inflight) < CONCURRENCY:
                pid = todo.pop(0)
                fut = ex.submit(run_trial, pid, env)
                inflight[fut] = pid
                log(f"dispatch {pid}  (inflight={len(inflight)})")

            time.sleep(5)

    scored = sum(1 for r in results.values() if is_done(r))
    total_cost = sum(float(r.get("cost_usd") or 0) for r in results.values())
    avg_reward = (
        sum(float(r.get("reward") or 0) for r in results.values() if is_done(r)) / scored
        if scored else 0.0
    )
    log(f"SWEEP COMPLETE. {scored}/{len(all_ids)} scored, "
        f"avg_reward={avg_reward:.4f}, total_cost=${total_cost:.2f}. "
        f"results -> {RESULTS}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Force codex onto HTTP (disable the Responses WebSocket transport).

Why this is needed
------------------
harbor's codex adapter, when OPENAI_BASE_URL is set, writes only
`openai_base_url = ...` into codex's config.toml. That merely overrides the
base_url of the *built-in* `openai` provider — and that provider has websockets
enabled. Against an HTTP-only custom node, codex then tries the Responses
WebSocket transport (which such nodes don't implement) and fails, even though
the key and REST endpoint are fine.

codex has no env var / CLI flag to disable websockets; the only lever is a
`supports_websockets = false` provider in config.toml, which only harbor's codex
adapter writes. So this script patches that adapter (in the installed harbor
package) to, when OPENAI_BASE_URL is set, write a dedicated `http_node` provider
with `supports_websockets = false` and select it — forcing plain HTTP.

`wire_api` (which HTTP endpoint) is chosen at run time via the CODEX_WIRE_API
env var: "responses" (POST /responses, default) or "chat" (POST /chat/completions,
what most OpenAI-compatible gateways implement). run.py forwards it via
--agent-env when set in .env.

Usage
-----
Run once AFTER `uv tool install harbor` (and re-run after upgrading harbor):

    python runs/codex-gpt55/patch_codex_http.py

Idempotent: safe to run repeatedly. Writes a .bak next to the patched file.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

MARKER = "model_providers.http_node"  # present iff already patched

# Exact block harbor 0.18.0 writes today (must match byte-for-byte).
OLD = r'''        # codex 0.118.0 only honors openai_base_url from config.toml, not the env var.
        config_toml_block = ""
        if openai_base_url:
            config_toml_block = (
                '\ncat >>"$CODEX_HOME/config.toml" <<TOML\n'
                'openai_base_url = "${OPENAI_BASE_URL}"\n'
                "TOML"
            )'''

NEW = r'''        # codex 0.118.0 only honors openai_base_url from config.toml, not the env var.
        # PATCHED by patch_codex_http.py: for a custom base_url, define a dedicated
        # provider with supports_websockets=false and select it, so codex uses HTTP
        # instead of the Responses WebSocket transport (HTTP-only nodes lack WS).
        # wire_api is chosen via CODEX_WIRE_API (responses|chat), default responses.
        config_toml_block = ""
        if openai_base_url:
            wire_api = (self._get_env("CODEX_WIRE_API") or "responses").strip().lower()
            if wire_api not in ("responses", "chat"):
                wire_api = "responses"
            config_toml_block = (
                '\ncat >>"$CODEX_HOME/config.toml" <<TOML\n'
                'openai_base_url = "${OPENAI_BASE_URL}"\n'
                'model_provider = "http_node"\n'
                "\n"
                "[model_providers.http_node]\n"
                'name = "http_node"\n'
                'base_url = "${OPENAI_BASE_URL}"\n'
                'env_key = "OPENAI_API_KEY"\n'
                f'wire_api = "{wire_api}"\n'
                "supports_websockets = false\n"
                "TOML"
            )'''


def find_codex_adapter() -> Path | None:
    patterns = [
        "~/.local/share/uv/tools/harbor/lib/python*/site-packages/harbor/agents/installed/codex.py",
        os.path.join(os.environ.get("XDG_DATA_HOME", "~/.local/share"),
                     "uv/tools/harbor/lib/python*/site-packages/harbor/agents/installed/codex.py"),
        "~/.local/pipx/venvs/harbor/lib/python*/site-packages/harbor/agents/installed/codex.py",
    ]
    for pat in patterns:
        hits = glob.glob(os.path.expanduser(pat))
        if hits:
            return Path(sorted(hits)[-1])
    # Fallback: importable harbor (e.g. pip-installed in a venv).
    try:
        import harbor  # type: ignore
        p = Path(harbor.__file__).parent / "agents" / "installed" / "codex.py"
        if p.is_file():
            return p
    except Exception:
        pass
    return None


def main() -> int:
    path = find_codex_adapter()
    if path is None:
        print("ERROR: could not locate harbor's codex adapter (codex.py).\n"
              "Install harbor first: `uv tool install harbor`.", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Already patched (found `{MARKER}`): {path}")
        return 0
    if OLD not in text:
        print(f"ERROR: expected config block not found in {path}\n"
              "harbor's codex adapter has changed — update OLD in this script, "
              "or apply the patch manually.", file=sys.stderr)
        return 3

    path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched {path}\n"
          "codex will now use a `supports_websockets=false` HTTP provider whenever "
          "OPENAI_BASE_URL is set (wire_api via CODEX_WIRE_API; default 'responses').\n"
          f"Backup written to {path}.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

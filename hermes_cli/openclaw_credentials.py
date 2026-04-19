"""Fallback credential bridge from OpenClaw's vault into Hermes env vars.

This module lets Hermes reuse secrets stored under ``~/.openclaw/credentials``
without copying them into ``~/.hermes/.env``.  Explicit process env vars and
values in ``~/.hermes/.env`` still win; this bridge only fills missing values.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

_OPENCLAW_POOL_INDEX = Path("~/clawd/config/credential_pool.json").expanduser()
_OPENCLAW_CREDENTIALS_DIR = Path("~/.openclaw/credentials").expanduser()

# env var -> ordered list of (OpenClaw credential file stem, candidate field names)
_BRIDGE_MAP: Dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "OPENROUTER_API_KEY": (("openrouter", ("openrouter_api_key", "api_key", "apiKey")),),
    "TAVILY_API_KEY": (("tavily", ("apiKey", "api_key")),),
    "MINIMAX_API_KEY": (("minimax", ("api_key", "apiKey")),),
    "GLM_API_KEY": (("zhipu", ("api_key", "apiKey")),),
    "ZAI_API_KEY": (("zhipu", ("api_key", "apiKey")),),
    "KIMI_API_KEY": (("moonshot", ("apiKey", "api_key")),),
    "XAI_API_KEY": (("xai", ("api_key", "apiKey")),),
    "JINA_API_KEY": (("jina", ("api_key", "apiKey")),),
    "TELEGRAM_BOT_TOKEN": (("hermes-telegram-bot", ("bot_token", "token")),),
}


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("OpenClaw credential bridge: failed to parse %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _candidate_paths(credential_name: str) -> Iterable[Path]:
    pool = _read_json(_OPENCLAW_POOL_INDEX) or {}
    providers = pool.get("providers") if isinstance(pool.get("providers"), dict) else {}
    provider_info = providers.get(credential_name) if isinstance(providers, dict) else None
    configured_path = None
    if isinstance(provider_info, dict):
        raw_path = provider_info.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            configured_path = Path(raw_path).expanduser()
    if configured_path is not None:
        yield configured_path
    yield _OPENCLAW_CREDENTIALS_DIR / f"{credential_name}.json"


def _load_single_value(credential_name: str, candidate_keys: Iterable[str]) -> Optional[str]:
    for path in _candidate_paths(credential_name):
        payload = _read_json(path)
        if not payload:
            continue
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def resolve_bridged_env_value(env_key: str) -> Optional[str]:
    for credential_name, candidate_keys in _BRIDGE_MAP.get(env_key, ()):
        value = _load_single_value(credential_name, candidate_keys)
        if value:
            return value
    return None


def get_all_bridged_env_values() -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for env_key in _BRIDGE_MAP:
        value = resolve_bridged_env_value(env_key)
        if value:
            resolved[env_key] = value
    return resolved


def populate_environment(force: bool = False) -> Dict[str, str]:
    """Populate os.environ with bridged values.

    By default, only fills variables that are currently unset.
    Returns the values applied during this call.
    """
    applied: Dict[str, str] = {}
    for env_key, value in get_all_bridged_env_values().items():
        if force or not os.environ.get(env_key):
            os.environ[env_key] = value
            applied[env_key] = value
    return applied

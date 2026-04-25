from __future__ import annotations

import os
import re
import sys
import threading
from typing import Any, Callable, Dict, Optional


_PATCH_LOCK = threading.Lock()
_PATCHED = False
_WARNED_MESSAGES: set[str] = set()


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _warn_once(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    print(f"[prompt-cache] {message}", file=sys.stderr)


def _sanitize_cache_key(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip())
    sanitized = sanitized.strip("-_.:")
    return sanitized[:128] or "lme"


def _build_prompt_cache_key(model: object) -> str:
    explicit_key = os.getenv("LME_PROMPT_CACHE_KEY", "").strip()
    if explicit_key:
        return _sanitize_cache_key(explicit_key)

    prefix = os.getenv("LME_PROMPT_CACHE_KEY_PREFIX", "lme-longmemeval")
    agent = os.getenv("LME_PROMPT_CACHE_AGENT", "agent")
    model_name = str(model or "model")
    return _sanitize_cache_key(f"{prefix}:{agent}:{model_name}")


def _cache_payload(kwargs: Dict[str, Any]) -> Dict[str, str]:
    if not _env_enabled("LME_PROMPT_CACHE_ENABLED", "1"):
        return {}
    if "prompt_cache_key" in kwargs:
        return {}

    payload = {"prompt_cache_key": _build_prompt_cache_key(kwargs.get("model"))}
    retention = os.getenv("LME_PROMPT_CACHE_RETENTION", "").strip()
    if retention:
        payload["prompt_cache_retention"] = retention
    return payload


def _merge_extra_body(kwargs: Dict[str, Any], payload: Dict[str, str]) -> Dict[str, Any]:
    merged = dict(kwargs)
    extra_body = dict(merged.get("extra_body") or {})
    extra_body.update(payload)
    merged["extra_body"] = extra_body
    return merged


def _looks_cache_related(exception: BaseException) -> bool:
    message = str(exception).lower()
    return "prompt_cache" in message or "prompt cache" in message


def _without_cache(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {
        key: value
        for key, value in kwargs.items()
        if key not in {"prompt_cache_key", "prompt_cache_retention"}
    }
    extra_body = dict(cleaned.get("extra_body") or {})
    extra_body.pop("prompt_cache_key", None)
    extra_body.pop("prompt_cache_retention", None)
    if extra_body:
        cleaned["extra_body"] = extra_body
    else:
        cleaned.pop("extra_body", None)
    return cleaned


def _with_key_only(kwargs: Dict[str, Any], payload: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if "prompt_cache_retention" not in payload:
        return None
    key_only_payload = {"prompt_cache_key": payload["prompt_cache_key"]}
    direct = _without_cache(kwargs)
    direct.update(key_only_payload)
    return direct


def _usage_value(usage: object, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    return int(getattr(usage, key, 0) or 0)


def _cached_tokens(usage: object) -> int:
    if usage is None:
        return 0
    details = None
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    else:
        details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return int(getattr(details, "cached_tokens", 0) or 0)


def _log_cache_usage(response: object, cache_key: str) -> None:
    if not _env_enabled("LME_PROMPT_CACHE_LOG", "1"):
        return
    usage = getattr(response, "usage", None)
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    cached_tokens = _cached_tokens(usage)
    if prompt_tokens or cached_tokens:
        print(
            f"[prompt-cache] key={cache_key} prompt_tokens={prompt_tokens} cached_tokens={cached_tokens}",
            file=sys.stderr,
        )


def _call_with_cache(
    original_create: Callable[..., Any],
    completion_resource: object,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    payload = _cache_payload(kwargs)
    if not payload:
        return original_create(completion_resource, *args, **kwargs)

    direct_kwargs = dict(kwargs)
    direct_kwargs.update(payload)
    cache_key = payload["prompt_cache_key"]

    try:
        response = original_create(completion_resource, *args, **direct_kwargs)
        _log_cache_usage(response, cache_key)
        return response
    except TypeError as exception:
        if not _looks_cache_related(exception):
            raise
        _warn_once("OpenAI SDK rejected direct prompt cache kwargs; retrying with extra_body.")
    except Exception as exception:
        if not _looks_cache_related(exception):
            raise
        key_only_kwargs = _with_key_only(direct_kwargs, payload)
        if key_only_kwargs is not None:
            _warn_once("Prompt cache retention was rejected; retrying with prompt_cache_key only.")
            try:
                response = original_create(completion_resource, *args, **key_only_kwargs)
                _log_cache_usage(response, cache_key)
                return response
            except Exception as key_only_exception:
                if not _looks_cache_related(key_only_exception):
                    raise
        _warn_once("OpenAI API rejected prompt cache parameters; retrying without cache hints.")
        return original_create(completion_resource, *args, **_without_cache(direct_kwargs))

    extra_body_kwargs = _merge_extra_body(kwargs, payload)
    try:
        response = original_create(completion_resource, *args, **extra_body_kwargs)
        _log_cache_usage(response, cache_key)
        return response
    except Exception as exception:
        if not _looks_cache_related(exception):
            raise
        key_only_kwargs = _with_key_only(extra_body_kwargs, payload)
        if key_only_kwargs is not None:
            _warn_once("Prompt cache retention was rejected via extra_body; retrying with prompt_cache_key only.")
            try:
                response = original_create(completion_resource, *args, **_merge_extra_body(kwargs, {"prompt_cache_key": cache_key}))
                _log_cache_usage(response, cache_key)
                return response
            except Exception as key_only_exception:
                if not _looks_cache_related(key_only_exception):
                    raise
        _warn_once("OpenAI SDK/API rejected prompt cache hints; retrying without cache hints.")
        return original_create(completion_resource, *args, **kwargs)


def install_openai_prompt_cache(default_agent: str) -> None:
    """Install a process-wide Chat Completions prompt-cache wrapper.

    OpenAI prompt caching is automatic; this wrapper only adds a stable
    prompt_cache_key to improve cache routing and logs cached token counts.
    """

    os.environ.setdefault("LME_PROMPT_CACHE_AGENT", default_agent)
    os.environ.setdefault("LME_PROMPT_CACHE_ENABLED", "1")
    os.environ.setdefault("LME_PROMPT_CACHE_KEY_PREFIX", "lme-longmemeval")
    os.environ.setdefault("LME_PROMPT_CACHE_LOG", "1")

    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        try:
            try:
                from openai.resources.chat.completions.completions import Completions
            except Exception:
                from openai.resources.chat.completions import Completions
        except Exception as exception:
            _warn_once(f"could not install OpenAI prompt cache wrapper: {exception}")
            return

        original_create = Completions.create

        def cached_create(completion_resource: object, *args: Any, **kwargs: Any) -> Any:
            return _call_with_cache(original_create, completion_resource, args, kwargs)

        setattr(cached_create, "_lme_prompt_cache_wrapped", True)
        Completions.create = cached_create
        _PATCHED = True

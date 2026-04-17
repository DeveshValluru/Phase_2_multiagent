"""vLLM OpenAI-compatible client with health checks + retry.

Addresses the Phase-1 issue: vLLM server going idle / dying mid-run.
- Pings /health before every request (cheap).
- Exponential backoff retries on transient failures.
- Background keepalive thread pings server during long gaps to prevent idle timeouts.
- Graceful SIGTERM handler so SLURM preemption doesn't corrupt state.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    max_tokens: int = 1024
    request_timeout: float = 300.0
    max_retries: int = 3
    backoff_seconds: float = 5.0
    strip_think: bool = False        # set True for Qwen3-32B


class VLLMClient:
    """Wraps the OpenAI client with health check + retry + keepalive."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.request_timeout,
            max_retries=0,   # we handle retries ourselves
        )
        self._health_url = cfg.base_url.rstrip("/").removesuffix("/v1") + "/health"
        self._stop_keepalive = threading.Event()
        self._keepalive_thread: threading.Thread | None = None
        self._term_requested = False
        self._install_signal_handlers()

    # ---- health & keepalive ------------------------------------------------

    def wait_ready(self, timeout_s: float = 900.0, poll_s: float = 5.0) -> None:
        """Block until /health responds 200, or raise after timeout."""
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._is_healthy():
                log.info("vLLM server ready at %s", self.cfg.base_url)
                return
            time.sleep(poll_s)
        raise RuntimeError(
            f"vLLM server did not become healthy within {timeout_s}s "
            f"(last error: {last_err}) — check {self._health_url}"
        )

    def _is_healthy(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(self._health_url)
                return r.status_code == 200
        except Exception:
            return False

    def start_keepalive(self, interval_s: float = 60.0) -> None:
        """Background thread pings /health every interval_s."""
        if self._keepalive_thread is not None:
            return

        def loop():
            while not self._stop_keepalive.is_set():
                self._is_healthy()
                self._stop_keepalive.wait(interval_s)

        self._stop_keepalive.clear()
        self._keepalive_thread = threading.Thread(
            target=loop, name="vllm-keepalive", daemon=True
        )
        self._keepalive_thread.start()

    def stop_keepalive(self) -> None:
        self._stop_keepalive.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=5.0)
            self._keepalive_thread = None

    # ---- signal handling ---------------------------------------------------

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            log.warning("received signal %s — setting term flag", signum)
            self._term_requested = True

        try:
            signal.signal(signal.SIGTERM, handler)
            signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError):
            # Signal handlers only work in main thread; ignore if not
            pass

    @property
    def term_requested(self) -> bool:
        return self._term_requested

    # ---- generation --------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Chat completion returning stripped text."""
        return self._chat_with_retry(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or self.cfg.max_tokens,
            stop=stop,
            response_format=response_format,
        )

    def _chat_with_retry(self, **kwargs) -> str:
        """Retry wrapper around a single chat request."""
        cfg = self.cfg

        @retry(
            reraise=True,
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential(multiplier=cfg.backoff_seconds, min=cfg.backoff_seconds, max=60.0),
            retry=retry_if_exception_type((
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                Exception,   # catch broad on purpose — vLLM can throw many
            )),
        )
        def _call() -> str:
            if not self._is_healthy():
                log.warning("vLLM /health failed — waiting and retrying")
                self.wait_ready(timeout_s=120.0)

            resp = self._client.chat.completions.create(
                model=cfg.model,
                messages=kwargs["messages"],
                temperature=kwargs["temperature"],
                top_p=kwargs["top_p"],
                max_tokens=kwargs["max_tokens"],
                stop=kwargs.get("stop"),
                extra_body=(
                    {"response_format": kwargs["response_format"]}
                    if kwargs.get("response_format") else None
                ),
            )
            text = resp.choices[0].message.content or ""
            if cfg.strip_think:
                from utils.parsing import strip_think
                text = strip_think(text)
            return text

        return _call()


# ---- convenience: build client from yaml section --------------------------

def client_from_config(cfg_section: dict[str, Any]) -> VLLMClient:
    """Build VLLMClient from a parsed YAML dict. Env vars expanded."""
    def expand(v):
        if isinstance(v, str):
            return os.path.expandvars(v)
        return v

    return VLLMClient(LLMConfig(
        base_url=expand(cfg_section["base_url"]),
        model=cfg_section["model"],
        api_key=str(cfg_section.get("api_key", "EMPTY")),
        max_tokens=int(cfg_section.get("max_tokens", 1024)),
        request_timeout=float(cfg_section.get("request_timeout", 300)),
        max_retries=int(cfg_section.get("max_retries", 3)),
        backoff_seconds=float(cfg_section.get("backoff_seconds", 5)),
        strip_think=bool(cfg_section.get("strip_think", False)),
    ))

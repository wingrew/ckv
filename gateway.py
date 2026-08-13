#!/usr/bin/env python3
"""Rank-0 OpenAI-compatible gateway for one or two SGLang replicas."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

LOG = logging.getLogger("metacamp.gateway")
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass
class Backend:
    url: str
    active: int = 0
    healthy: bool = True
    consecutive_failures: int = 0
    registered_at: float = 0.0


class BackendRouter:
    def __init__(self, per_backend_limit: int) -> None:
        self.per_backend_limit = per_backend_limit
        self.backends: dict[str, Backend] = {}
        self.condition = asyncio.Condition()
        self.rr_cursor = 0

    @staticmethod
    def normalize_url(url: str) -> str:
        value = url.rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
            raise ValueError("backend URL must look like http://HOST:PORT")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("backend URL must not contain a path, query, or fragment")
        return value

    async def register(self, url: str) -> Backend:
        url = self.normalize_url(url)
        async with self.condition:
            backend = self.backends.get(url)
            if backend is None:
                backend = Backend(url=url, registered_at=time.time())
                self.backends[url] = backend
                LOG.info("registered backend %s", url)
            else:
                backend.healthy = True
                backend.consecutive_failures = 0
                LOG.info("refreshed backend registration %s", url)
            self.condition.notify_all()
            return backend

    async def acquire(self, excluded_urls: set[str] | None = None) -> Backend:
        """Wait until a healthy, non-excluded backend has a free request slot."""
        excluded_urls = excluded_urls or set()
        async with self.condition:
            while True:
                candidates = [
                    backend
                    for backend in self.backends.values()
                    if backend.url not in excluded_urls
                    and backend.healthy
                    and backend.active < self.per_backend_limit
                ]
                if candidates:
                    minimum = min(item.active for item in candidates)
                    least_loaded = [item for item in candidates if item.active == minimum]
                    backend = least_loaded[self.rr_cursor % len(least_loaded)]
                    self.rr_cursor += 1
                    backend.active += 1
                    return backend
                await self.condition.wait()

    async def release(self, backend: Backend) -> None:
        async with self.condition:
            current = self.backends.get(backend.url)
            if current is not None:
                current.active = max(0, current.active - 1)
            self.condition.notify_all()

    async def record_health(
        self, url: str, success: bool, *, immediate_failure: bool = False
    ) -> None:
        async with self.condition:
            backend = self.backends.get(url)
            if backend is None:
                return
            old_healthy = backend.healthy
            if success:
                backend.consecutive_failures = 0
                backend.healthy = True
            else:
                backend.consecutive_failures += 1
                if immediate_failure or backend.consecutive_failures >= 3:
                    backend.healthy = False
            if old_healthy != backend.healthy:
                LOG.warning("backend %s healthy=%s", url, backend.healthy)
            self.condition.notify_all()

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self.condition:
            return [
                {
                    "url": item.url,
                    "active": item.active,
                    "limit": self.per_backend_limit,
                    "healthy": item.healthy,
                    "consecutive_failures": item.consecutive_failures,
                    "registered_at": item.registered_at,
                }
                for item in self.backends.values()
            ]


class ContractError(ValueError):
    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


def is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_and_normalize_payload(payload: Any) -> dict[str, Any]:
    """Validate fixed-output controls and preserve all other request fields."""
    if not isinstance(payload, dict):
        raise ContractError("request body must be a JSON object")

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise ContractError("model must be a string", "model")

    for field in ("max_tokens", "max_completion_tokens"):
        if field in payload and payload[field] is not None:
            if not is_positive_integer(payload[field]):
                raise ContractError(f"{field} must be a positive integer", field)

    if "min_tokens" in payload:
        if not is_positive_integer(payload["min_tokens"]):
            raise ContractError("min_tokens must be a positive integer", "min_tokens")

    if "ignore_eos" in payload and not isinstance(payload["ignore_eos"], bool):
        raise ContractError("ignore_eos must be a boolean", "ignore_eos")

    max_tokens = payload.get("max_tokens")
    max_completion_tokens = payload.get("max_completion_tokens")

    # SGLang gives max_completion_tokens precedence. Keep max_tokens a hard upper
    # bound even when a client sends both fields by forwarding the smaller limit.
    effective_max: int | None = None
    if max_tokens is not None and max_completion_tokens is not None:
        effective_max = min(max_tokens, max_completion_tokens)
        payload["max_tokens"] = effective_max
        payload["max_completion_tokens"] = effective_max
    elif max_tokens is not None:
        effective_max = max_tokens
    elif max_completion_tokens is not None:
        effective_max = max_completion_tokens

    min_tokens = payload.get("min_tokens")
    if min_tokens is not None and effective_max is not None and min_tokens > effective_max:
        raise ContractError(
            f"min_tokens ({min_tokens}) cannot exceed the effective maximum ({effective_max})",
            "min_tokens",
        )

    return payload


def openai_error(message: str, status_code: int, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": param,
                "code": status_code,
            }
        },
    )


def forwarded_request_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    headers["content-type"] = "application/json"
    return headers


def forwarded_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def create_app(args: argparse.Namespace) -> FastAPI:
    router = BackendRouter(args.backend_limit)
    client: httpx.AsyncClient | None = None
    health_task: asyncio.Task[None] | None = None

    async def health_loop() -> None:
        assert client is not None
        while True:
            snapshot = await router.snapshot()
            for item in snapshot:
                url = item["url"]
                success = False
                try:
                    response = await client.get(f"{url}/health", timeout=2.0)
                    success = response.status_code < 500
                except (httpx.HTTPError, asyncio.TimeoutError):
                    success = False
                await router.record_health(url, success)
            await asyncio.sleep(args.health_interval)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal client, health_task
        limits = httpx.Limits(
            max_connections=max(128, args.backend_limit * args.expected_backends * 2),
            max_keepalive_connections=max(64, args.backend_limit * args.expected_backends),
            keepalive_expiry=30.0,
        )
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=10.0),
            limits=limits,
            follow_redirects=False,
        )
        await router.register(args.backend)
        health_task = asyncio.create_task(health_loop(), name="backend-health-loop")
        try:
            yield
        finally:
            if health_task is not None:
                health_task.cancel()
                try:
                    await health_task
                except asyncio.CancelledError:
                    pass
            await client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        snapshot = await router.snapshot()
        healthy = sum(1 for item in snapshot if item["healthy"])
        status = 200 if healthy > 0 else 503
        return JSONResponse(
            status_code=status,
            content={
                "status": "ok" if status == 200 else "unavailable",
                "healthy_backends": healthy,
                "expected_backends": args.expected_backends,
                "backends": snapshot,
            },
        )

    @app.get("/ready")
    async def ready() -> JSONResponse:
        snapshot = await router.snapshot()
        healthy = sum(1 for item in snapshot if item["healthy"])
        status = 200 if healthy >= args.expected_backends else 503
        return JSONResponse(
            status_code=status,
            content={
                "ready": status == 200,
                "healthy_backends": healthy,
                "expected_backends": args.expected_backends,
            },
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "GLM-5.2",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "metacamp",
                }
            ],
        }

    @app.post("/_internal/register")
    async def register_backend(request: Request) -> Response:
        supplied = request.headers.get("x-metacamp-registration-token", "")
        if not secrets.compare_digest(supplied, args.registration_token):
            return openai_error("invalid registration token", 403)
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("url"), str):
                raise ValueError("body must contain a string field named 'url'")
            backend = await router.register(body["url"])
        except (ValueError, json.JSONDecodeError) as exc:
            return openai_error(str(exc), 400, "url")
        return JSONResponse({"registered": backend.url})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        assert client is not None
        try:
            raw_body = await request.body()
            payload = json.loads(raw_body)
            payload = validate_and_normalize_payload(payload)
        except json.JSONDecodeError:
            return openai_error("request body is not valid JSON", 400)
        except ContractError as exc:
            return openai_error(str(exc), 400, exc.param)

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request_headers = forwarded_request_headers(request)
        backend: Backend | None = None
        response: httpx.Response | None = None
        tried: set[str] = set()
        last_error = "no backend was available"

        for attempt in range(max(1, args.expected_backends)):
            try:
                if attempt == 0:
                    backend = await router.acquire()
                else:
                    backend = await asyncio.wait_for(
                        router.acquire(excluded_urls=tried),
                        timeout=args.failover_wait,
                    )
            except asyncio.TimeoutError:
                last_error = "timed out waiting for a failover backend"
                break

            try:
                outbound = client.build_request(
                    "POST",
                    f"{backend.url}/v1/chat/completions",
                    content=body,
                    headers=request_headers,
                )
                response = await client.send(outbound, stream=True)
                if response.status_code >= 500:
                    last_error = f"backend {backend.url} returned HTTP {response.status_code}"
                    await response.aclose()
                    response = None
                    tried.add(backend.url)
                    await router.record_health(
                        backend.url, False, immediate_failure=True
                    )
                    await router.release(backend)
                    backend = None
                    continue
                await router.record_health(backend.url, True)
                break
            except httpx.HTTPError as exc:
                last_error = f"backend {backend.url} request failed: {exc}"
                LOG.exception("backend request failed: %s", backend.url)
                tried.add(backend.url)
                await router.record_health(
                    backend.url, False, immediate_failure=True
                )
                await router.release(backend)
                backend = None

        if backend is None or response is None:
            return openai_error(last_error, 502)

        response_headers = forwarded_response_headers(response)
        is_stream = payload.get("stream") is True

        if is_stream:
            async def relay() -> Any:
                try:
                    async for chunk in response.aiter_raw():
                        yield chunk
                finally:
                    await response.aclose()
                    await router.release(backend)

            return StreamingResponse(
                relay(),
                status_code=response.status_code,
                headers=response_headers,
                media_type=None,
            )

        try:
            content = await response.aread()
            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers,
            )
        finally:
            await response.aclose()
            await router.release(backend)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", required=True, help="rank-0 local SGLang URL")
    parser.add_argument("--backend-limit", type=int, default=32)
    parser.add_argument("--expected-backends", type=int, choices=(1, 2), required=True)
    parser.add_argument("--registration-token", required=True)
    parser.add_argument("--health-interval", type=float, default=2.0)
    parser.add_argument("--failover-wait", type=float, default=30.0)
    args = parser.parse_args()
    if args.backend_limit <= 0:
        parser.error("--backend-limit must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.failover_wait <= 0:
        parser.error("--failover-wait must be positive")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    app = create_app(args)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()

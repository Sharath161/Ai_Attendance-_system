"""Observability: request IDs, timing headers and Prometheus counters."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict

from fastapi import Request

METRICS = {"requests": defaultdict(int), "errors": defaultdict(int),
           "latency_sum": defaultdict(float)}


async def observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        METRICS["errors"][request.url.path] += 1
        raise
    elapsed = time.perf_counter() - started
    endpoint = request.url.path
    METRICS["requests"][endpoint] += 1
    METRICS["latency_sum"][endpoint] += elapsed
    if response.status_code >= 400:
        METRICS["errors"][endpoint] += 1
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed * 1000:.1f}"
    return response


def prometheus_text() -> str:
    lines = ["# HELP faceapi_requests_total Requests per endpoint",
             "# TYPE faceapi_requests_total counter"]
    lines += [f'faceapi_requests_total{{endpoint="{e}"}} {n}'
              for e, n in METRICS["requests"].items()]
    lines += [f'faceapi_errors_total{{endpoint="{e}"}} {n}'
              for e, n in METRICS["errors"].items()]
    lines += [f'faceapi_request_duration_seconds_sum{{endpoint="{e}"}} {s:.4f}'
              for e, s in METRICS["latency_sum"].items()]
    return "\n".join(lines) + "\n"

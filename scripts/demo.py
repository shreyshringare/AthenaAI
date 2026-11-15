"""Narrated AthenaAI demo — shows each headline feature happening live.

Run the server in demo mode first (two controllable mock backends, free, no
API keys):

    docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --build
    python scripts/demo.py            # add --fast to skip the pauses

Steps: 1 complexity-tier routing, 2 retry + circuit breaker + failover,
3 rate limiting (429), 4 agent with parallel tool calls. Open Grafana at
http://localhost:3000 (admin / athena) alongside to watch the metrics move.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import httpx

BASE = "http://localhost:8000"
B, G, Y, R, C, D, X = "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[2m", "\033[0m"  # noqa: E501

client = httpx.Client(base_url=BASE, timeout=30)
pause_s = 1.5


def step(n: int, title: str) -> None:
    print(f"\n{B}{C}━━ {n}. {title} ━━{X}")
    time.sleep(pause_s)


def chat(content: str, key: str | None = None) -> httpx.Response:
    headers = {"x-api-key": key} if key else {}
    return client.post(
        "/v1/chat",
        json={"messages": [{"role": "user", "content": content}], "session_id": f"demo-{uuid.uuid4()}"},  # noqa: E501
        headers=headers,
    )


def status() -> dict:
    return client.get("/debug/status").json()


def show_status(s: dict) -> None:
    for role, info in s.items():
        color = G if info["circuit"] == "CLOSED" else R
        down = f" {R}(failing){X}" if info["failing"] else ""
        print(f"   {role:<11} circuit {color}{info['circuit']}{X}{down}")


def main() -> int:
    global pause_s
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="no pauses between steps")
    pause_s = 0 if ap.parse_args().fast else 1.5

    try:
        s = client.get("/debug/status")
    except httpx.ConnectError:
        print(f"{R}Server not reachable at {BASE}.{X} Start it first (see this file's docstring).")
        return 1
    if s.status_code == 404:
        print(f"{R}Server is not in demo mode.{X} Start with docker-compose.demo.yml.")
        return 1
    client.post("/debug/reset")

    print(f"{B}AthenaAI demo{X} {D}— mock backends: fast-mock (cheap, quick, ok quality) "
          f"vs smart-mock (pricey, slow, best){X}")

    # 1 ── routing ──────────────────────────────────────────────────────────
    step(1, "Routing by token-count complexity tier")
    short = chat("What is the capital of France?").json()
    print(f"   short prompt  (~10 tokens,  LOW tier)  → served by {G}{short['model']}{X}")
    long_prompt = "Summarise this report. " + "Lorem ipsum dolor sit amet. " * 400
    long = chat(long_prompt).json()
    print(f"   long prompt   (~{long['usage']['input_tokens']} tokens, HIGH tier) → "
          f"served by {G}{long['model']}{X}")
    print(f"   {D}weights: quality/cost/latency scored per model, tilted by tier{X}")

    # 2 ── resilience ───────────────────────────────────────────────────────
    step(2, "Retry + circuit breaker + failover")
    print("   healthy:")
    show_status(status())
    client.post("/debug/fail", json={"role": "fast-mock", "failing": True})
    print(f"\n   {Y}fast-mock just went down.{X} Sending a short prompt (normally → fast-mock)…")
    t = time.perf_counter()
    r = chat("Still there?").json()
    print(f"   → {G}answered by {r['model']}{X} in {time.perf_counter() - t:.2f}s "
          f"{D}(3 retries w/ backoff+jitter on fast-mock, then fell over){X}")
    show_status(status())
    t = time.perf_counter()
    r = chat("And again?").json()
    print(f"   next request → {G}{r['model']}{X} in {time.perf_counter() - t:.2f}s "
          f"{D}(circuit OPEN: fast-mock skipped at routing time, no retries){X}")
    client.post("/debug/reset")
    print(f"   {D}reset: backends healed, circuits closed{X}")

    # 3 ── rate limiting ────────────────────────────────────────────────────
    step(3, "Per-client rate limiting (token bucket)")
    sent = 0
    resp = None
    for _ in range(500):
        resp = chat("burst", key="demo-burst-client")
        sent += 1
        if resp.status_code == 429:
            break
    if resp is not None and resp.status_code == 429:
        print(f"   {sent - 1} requests accepted, then {R}HTTP 429{X} "
              f"(Retry-After: {resp.headers.get('retry-after')}s)")
    else:
        print(f"   {Y}no 429 after {sent} requests — rate limit may be disabled{X}")
    other = chat("hello", key="someone-else")
    print(f"   a different client is unaffected → HTTP {G}{other.status_code}{X}")

    # 4 ── agent ────────────────────────────────────────────────────────────
    step(4, "Agent: state machine + parallel tool calls")
    a = client.post(
        "/v1/agents/run", json={"task": "calc: 12*7; 2**10; 100/4; (3+4)*5"}
    ).json()
    for st in a["steps"]:
        calls = [c["arguments"]["expression"] for c in st["tool_calls"]]
        if calls:
            print(f"   iteration {st['iteration']}: {len(calls)} tool calls run in parallel "
                  f"via asyncio.gather → {[r.get('result') for r in st['tool_results']]}")
        else:
            print(f"   iteration {st['iteration']}: final answer")
    print(f"   {G}{a['status']}{X} in {a['total_iterations']} iterations "
          f"{D}(hard-capped by max_iterations){X} → {a['final_answer']}")

    print(f"\n{B}Done.{X} Metrics: {BASE}/metrics · Swagger: {BASE}/docs · "
          f"Grafana: http://localhost:3000")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Live integration check for the Vizhi Python SDK.

This script verifies the end-to-end path:

1. SDK sends the Vizhi token in the Authorization header.
2. Backend validates the token.
3. Backend routes the request to the configured provider/final_call backend.
4. SDK parses the response metadata.
"""

from __future__ import annotations

import os
import sys

import vizhi_sdk as vs


def main() -> int:
    token = os.environ.get("VIZHI_API_TOKEN", "").strip() or "vz_live_94982905f68f31bf3e653661da1062def9b8d4c6a1c58add"
    base_url = os.environ.get("VIZHI_BASE_URL", "http://localhost:8000").strip()
    model = os.environ.get("VIZHI_MODEL", "").strip() or None
    prompt = os.environ.get(
        "VIZHI_TEST_PROMPT",
        "Who is Sundar Pichai?",
    )

    if not token:
        print("Missing VIZHI_API_TOKEN. Export your frontend-generated token first.")
        return 2

    print("Running Vizhi live integration check...")
    print(f"Backend: {base_url}")
    print(f"Model: {model or '(using model bound to token)'}")

    try:
        provider = vs.provide_model(
            model,
            token,
            base_url=base_url,
            timeout=120,
        )
        answer = provider.chat(prompt, temperature=0.2, max_tokens=80)
    except Exception as exc:
        print("Integration check failed.")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    print("Integration check passed.")
    print(f"Answer: {answer.content}")
    print(f"Provider: {answer.provider}")
    print(f"Model: {answer.model}")
    print(f"Agent ID: {answer.agent_id}")
    print(f"Query ID: {answer.query_id}")
    print(f"Input tokens: {answer.input_tokens}")
    print(f"Output tokens: {answer.output_tokens}")
    print(f"Latency ms: {answer.latency_ms}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

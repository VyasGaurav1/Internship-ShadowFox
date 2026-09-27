"""Route doctor for the Student Utility App.

Prints exactly which provider/route the app will use (with the key masked) and,
with ``--live``, sends one real request through that route.

    python doctor.py           # show the resolved configuration
    python doctor.py --live    # configuration + one real request
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def masked(value: str | None) -> str:
    if not value:
        return "<missing>"
    return value[:6] + "..." + value[-4:] if len(value) > 12 else "<set>"


def main() -> int:
    from config import load_settings
    from llm_client import LLMClient, LLMError

    settings = load_settings()
    print("Resolved configuration")
    print(f"  config file      : {os.path.join(HERE, '.env')}  (exists: {os.path.exists(os.path.join(HERE, '.env'))})")
    print(f"  LLM_PROVIDER     : {settings.provider}")
    print(f"  base_url         : {settings.base_url or '<none>'}")
    print(f"  model            : {settings.model}")
    print(f"  credential       : {masked(settings.credential)}")
    print(f"  offline fallback : {settings.is_offline}")

    if "--live" not in sys.argv:
        print("\nAdd --live to send one real request through this route.")
        return 0

    print("\nSending one live request ...")
    try:
        result = LLMClient(settings).complete(
            "summarize",
            "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs light. "
            "The Calvin cycle fixes carbon dioxide into glucose.",
        )
    except LLMError as exc:
        print(f"  FAILED ({exc.kind}): {exc.message}")
        return 1

    print(f"  route used       : {result.provider}:{result.model}")
    print(f"  offline          : {result.offline}")
    print(f"  latency          : {result.elapsed_ms} ms")
    print("  response head    : " + result.text.strip().replace("\n", " ")[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

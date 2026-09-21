"""Serve the dependency-free presentation demo.

This is a UI rehearsal and integration demonstrator, not a trained detector.
It deliberately labels every result as demonstration data.

Run from the repository root:
    python scripts/presentation_demo.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "results" / "presentation_demo" / "status.json"


def write_status() -> None:
    out = STATUS_PATH.parent
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "mode": "presentation_demo",
        "trained_model_available": False,
        "real_dataset_available": False,
        "real_metrics_available": False,
        "warning": "Demonstration outputs are deterministic UI fixtures and must not be reported as trained-model results.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the dependency-free presentation demo")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    write_status()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    print(f"Presentation demo: http://127.0.0.1:{args.port}/dashboard/presentation_demo.html")
    print("DEMO MODE: outputs are deterministic fixtures, not trained-model results.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPresentation demo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
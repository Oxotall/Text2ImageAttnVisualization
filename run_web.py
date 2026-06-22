"""Launch the attention-visualization web app.

Example:
    python run_web.py --port 8000
Then open http://127.0.0.1:8000 in your browser.

Note: port 5000 is avoided by default because on macOS it is used by the
AirPlay Receiver (Control Center), which returns "HTTP 403 Forbidden".
"""

from __future__ import annotations

import argparse

from attnviz import Config
from webapp import VizServer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the attention visualizer web UI.")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    p.add_argument("--port", type=int, default=8000, help="Port to bind.")
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset).")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   help="Default HuggingFace model id (also switchable in the UI).")
    p.add_argument("--debug", action="store_true", help="Flask debug mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(model_id=args.model, device=args.device)
    server = VizServer(config)
    print(f"Serving on http://{args.host}:{args.port}  (model loads on first generate)")
    server.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

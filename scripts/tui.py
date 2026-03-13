import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from art.tui.app import ArtApp


def main():
    parser = argparse.ArgumentParser(description="ArtAgent TUI - Beautiful evolution viewer")
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to run (default: 50)",
    )
    parser.add_argument(
        "--resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Resume from latest checkpoint (default: True)",
    )
    parser.add_argument(
        "--vlm",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use local VLM (moondream via ollama) as critic (default: False)",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="moondream",
        help="Ollama model name for VLM critic (default: moondream)",
    )
    parser.add_argument(
        "--web",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Start WebSocket bridge for browser visualization (default: False)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8765,
        help="WebSocket port for --web mode (default: 8765)",
    )

    args = parser.parse_args()

    app = ArtApp(
        generations=args.generations,
        resume=args.resume,
        use_vlm=args.vlm,
        vlm_model=args.vlm_model,
        web=args.web,
        web_port=args.web_port,
    )
    app.run()


if __name__ == "__main__":
    main()

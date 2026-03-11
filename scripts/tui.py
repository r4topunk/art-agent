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

    args = parser.parse_args()

    app = ArtApp(generations=args.generations, resume=args.resume)
    app.run()


if __name__ == "__main__":
    main()

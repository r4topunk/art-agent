import argparse
import sys
from pathlib import Path

# Add parent directory to path to import art module
sys.path.insert(0, str(Path(__file__).parent.parent))

from art.config import ArtConfig
from art.runner import OvernightRunner
from art.utils import ensure_dirs


def main():
    parser = argparse.ArgumentParser(description="Run overnight evolution of ArtAgent")
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
        help="Resume from latest checkpoint if available (default: True)",
    )

    args = parser.parse_args()

    # Create config and ensure directories
    config = ArtConfig()
    ensure_dirs(config)

    # Create runner
    runner = OvernightRunner(config)

    try:
        if args.resume:
            if not runner.resume():
                print("No previous generation found. Starting fresh.")
                runner.initialize()
        else:
            print("--no-resume specified. Starting fresh.")
            runner.initialize()

        runner.run(args.generations)

    except KeyboardInterrupt:
        generation_num = runner.gas.generation if hasattr(runner, 'gas') else 0
        print(f"\nInterrupted. Progress saved up to generation {generation_num}.")
        sys.exit(0)


if __name__ == "__main__":
    main()

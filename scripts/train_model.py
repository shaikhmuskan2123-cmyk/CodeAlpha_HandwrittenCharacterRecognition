"""Train the full EMNIST model, or use --smoke for a tiny pipeline check."""

import argparse

from src.training import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train for two epochs on a tiny balanced subset; not for evaluation or the app",
    )
    model_path = train(parser.parse_args().smoke)
    print(f"Saved model: {model_path}")
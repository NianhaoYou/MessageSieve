from __future__ import annotations

import argparse
import json

from .config import load_eval_config
from .runner import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MessageSieve evaluation config.")
    parser.add_argument("--config", required=True, help="Path to the evaluation YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_eval_config(args.config)
    result = run_eval(config)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

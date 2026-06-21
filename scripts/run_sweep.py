import argparse
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def inject_tags(config: dict, extra_tags: list[str]) -> dict:
    existing = config.get("tags", [])
    merged = existing + [t for t in extra_tags if t not in existing]
    config["tags"] = merged
    return config


def inject_sweep_name(config: dict, sweep_name: str) -> dict:
    output = config.setdefault("output", {})
    output["sweep_name"] = sweep_name
    return config


def run_sweep(config_paths: list[str], extra_tags: list[str], sweep_name: str | None) -> None:
    from run_train import train_config

    expanded = []
    for pattern in config_paths:
        matches = sorted(Path().glob(pattern))
        if matches:
            expanded.extend(str(m) for m in matches)
        else:
            expanded.append(pattern)

    results = []
    n = len(expanded)

    print(f"\nRunning sweep of {n} config(s)")
    if extra_tags:
        print(f"Extra tags: {extra_tags}")
    if sweep_name:
        print(f"Sweep name: {sweep_name}")

    for idx, config_path in enumerate(expanded, start=1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{n}] {config_path}")
        print(f"{'='*60}")

        config = load_config(config_path)
        config = inject_tags(config, extra_tags)
        if sweep_name:
            config = inject_sweep_name(config, sweep_name)

        result = train_config(config)
        results.append((config_path, result))

    print(f"\n{'='*60}")
    print("Sweep complete")
    print(f"{'='*60}")
    for config_path, result in results:
        f1_key = "best_dev_f1" if "best_dev_f1" in result else "macro_f1"
        f1 = result.get(f1_key, 0.0)
        print(f"  {config_path}: {f1_key}={f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sweep over multiple training configs")
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="One or more config YAML paths",
    )
    parser.add_argument(
        "--tags", nargs="*", default=[],
        help="Extra W&B tags appended to every run",
    )
    parser.add_argument(
        "--sweep-name", type=str, default=None,
        help="Sweep name for grouping checkpoints (injected as output.sweep_name)",
    )
    args = parser.parse_args()
    run_sweep(args.configs, args.tags, args.sweep_name)


if __name__ == "__main__":
    main()
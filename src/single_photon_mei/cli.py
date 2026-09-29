from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

from .stage1 import finalize_stage1, run_stage1, validate_stage1_config
from .stage2 import final_report, generate_mei, peer_review, select_targets, validate_stage2_config
from .utils.io import load_config, project_paths, sha256_file, validate_neural_data


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Offline standalone single-photon MEI workflow")
    sub = root.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)

    stage1 = sub.add_parser("stage1")
    stage1_sub = stage1.add_subparsers(dest="step", required=True)
    run = stage1_sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--model", action="append", default=None)
    run.add_argument("--overwrite", action="store_true")
    finalize = stage1_sub.add_parser("finalize")
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--overwrite", action="store_true")

    stage2 = sub.add_parser("stage2")
    stage2_sub = stage2.add_subparsers(dest="step", required=True)
    for name in ("select", "generate", "peer-review", "report"):
        command = stage2_sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name != "generate":
            command.add_argument("--overwrite", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        if config.get("schema_version") == "single-photon-mei-stage1-1":
            validate_stage1_config(config)
            weights = project_paths(config)["weights"]
            for key, spec in config["models"].items():
                checkpoint = weights / spec["checkpoint"]
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                if sha256_file(checkpoint) != spec["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {key}")
        elif config.get("schema_version") == "single-photon-mei-stage2-1":
            validate_stage2_config(config)
        else:
            raise ValueError("Unknown config schema")
        manifest, qc = validate_neural_data(config)
        result = {**qc, "manifest_rows": len(manifest)}
    elif args.command == "stage1" and args.step == "run":
        result = run_stage1(config, args.model, args.overwrite)
    elif args.command == "stage1":
        result = finalize_stage1(config, args.overwrite)
    elif args.step == "select":
        result = select_targets(config, args.overwrite)
    elif args.step == "generate":
        result = generate_mei(config)
    elif args.step == "peer-review":
        result = peer_review(config, args.overwrite)
    else:
        result = final_report(config, args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

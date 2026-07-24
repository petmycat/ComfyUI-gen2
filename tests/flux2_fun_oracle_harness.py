from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch

from flux2_fun.oracle import ORACLE_TENSOR_NAMES, REQUIRED_ORACLE_TENSOR_NAMES, compare_tensors, write_oracle_report


def load_bundle(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or "tensors" not in payload:
        raise ValueError(f"Unsupported Flux2 Fun oracle bundle: {path}")
    tensor_names = set(payload["tensors"])
    unknown = sorted(tensor_names - REQUIRED_ORACLE_TENSOR_NAMES)
    if unknown:
        raise ValueError(f"Unknown tensor names in {path}: {unknown}")
    missing = sorted(REQUIRED_ORACLE_TENSOR_NAMES - tensor_names)
    if missing:
        raise ValueError(f"Missing required tensor names in {path}: {missing}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Compare Gen2 Flux2 Fun tensors with a pinned VideoX-Fun oracle bundle.")
    parser.add_argument("--actual", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    if os.environ.get("GEN2_FLUX2_FUN_ORACLE") != "1":
        raise SystemExit("Set GEN2_FLUX2_FUN_ORACLE=1 to run the optional real-weight parity harness.")

    actual = load_bundle(args.actual)
    reference = load_bundle(args.reference)
    comparisons = [
        compare_tensors(name, actual["tensors"][name], reference["tensors"][name], atol=args.atol, rtol=args.rtol)
        for name in ORACLE_TENSOR_NAMES
    ]
    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "compared_count": len(comparisons),
        "actual_metadata": actual.get("metadata", {}),
        "reference_metadata": reference.get("metadata", {}),
    }
    write_oracle_report(args.report, comparisons, metadata)
    print(json.dumps({"report": str(Path(args.report).resolve()), "passed": all(item.passed for item in comparisons)}))
    raise SystemExit(0 if all(item.passed for item in comparisons) else 1)


if __name__ == "__main__":
    main()

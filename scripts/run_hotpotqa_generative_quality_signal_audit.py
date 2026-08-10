#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragtune.hotpotqa_generative_validation import run_hotpotqa_quality_signal_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/ragtune_hotpotqa_generative_quality_signal_audit_v1.yaml")
    parser.add_argument("--output-root", default="artifacts/generative_llm_validation/hotpotqa_quality_signal_audit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = run_hotpotqa_quality_signal_audit(ROOT, output_root=ROOT / args.output_root, dry_run=args.dry_run)
    print(f"HotpotQA generative quality-signal audit: {payload['result_class']} / {payload['quality_signal_audit_result_class']}")


if __name__ == "__main__":
    main()

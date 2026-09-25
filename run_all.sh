#!/usr/bin/env bash
# Reproduces every number in PAPER.md. About 5 minutes on one RTX 3090.
# Set XT_MODEL to use a different model (default: Qwen/Qwen3-1.7B-Base).
set -e
cd "$(dirname "$0")"
for s in bits.py roofline.py determinism.py kvcache.py attribution.py; do
    echo "### $s"
    python3 "$s"
    echo
done

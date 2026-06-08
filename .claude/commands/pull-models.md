---
description: Download the three models (FP8 Qwen-VL + Llama, BGE) into the HF cache
---

Download the models this project needs by running `rocm/pull-models.sh` in the
background and report progress. The weights live on the ephemeral disk (~153 GB),
so this is also the per-session re-pull. No HF token is required (all ungated).

Steps:
1. Start: `nohup bash rocm/pull-models.sh > /workspace/shared/.logs/model-pull.log 2>&1 &`
2. Tail the log and summarize which of qwen / llama / bge are done vs in progress.
3. Report total cache size with `du -sh ~/.cache/huggingface/hub`.

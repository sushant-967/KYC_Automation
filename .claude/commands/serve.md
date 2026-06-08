---
description: Launch the three vLLM servers (Qwen-VL :8000, Llama :8001, BGE :8002)
---

Bring up the model servers on this box. Requires the weights to be present already
(`/pull-models` first). Launch each in the background, then verify.

Steps:
1. Confirm the GPU is free: `rocm-smi --showmeminfo vram`.
2. Launch each server in the background, logging to `/workspace/shared/.logs/`:
   - `nohup bash rocm/vllm-launch-bge.sh   > /workspace/shared/.logs/vllm-bge.log   2>&1 &`
   - `nohup bash rocm/vllm-launch-llama.sh > /workspace/shared/.logs/vllm-llama.log 2>&1 &`
   - `nohup bash rocm/vllm-launch-qwen.sh  > /workspace/shared/.logs/vllm-qwen.log  2>&1 &`
3. Poll until each `curl -s http://localhost:PORT/v1/models` returns its model id
   (the 70B/72B models take a few minutes to load). Report readiness per port and
   surface any errors from the logs.

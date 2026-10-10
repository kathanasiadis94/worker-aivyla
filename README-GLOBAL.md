# Aivyla Global — isolated model cache (test build)

This **does not change the production image** `ghcr.io/kathanasiadis94/worker-aivyla:aivyla-5`. The new worker is a thin image *on top of* the existing image and must be deployed **only** on endpoint **Aivyla Global**.

## Files

- `Dockerfile.global` — standalone isolated image from `aivyla-5`.
- `aivyla_global_start.py` — preserves original GPU/Sage/ComfyUI boot checks and inserts local cache paths with highest resolution priority.
- `aivyla_global_nodes.py` — copies 16 custom nodes' Python/source locally; large model assets remain linked to Global Volume. No writes to source.
- `aivyla_global_cache.py` — detects file-based model references in each workflow, copies only those weights locally, streams SHA256 verification, validates safetensors header and checks disk. Cache is local to the current worker.
- `aivyla_global_handler.py` — wraps the installed handler without modifying existing logic or diagnostics. It stages weights before sending the prompt to ComfyUI.
- `.github/workflows/build-aivyla-global.yml` — optional manual GH Actions build; uses a distinct image tag.
- `test_global.py` — local tests (6 passing without GPU).

## Source / build

Copy these files into the **root of a checkout** of `kathanasiadis94/worker-aivyla` at branch `aivyla`, preserving `.github/workflows/`. Do NOT replace `Dockerfile`, `handler.py`, or `aivyla/aivyla-start.sh`.

Build from an amd64-compatible Docker setup, authenticated to GHCR:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.global \
  -t ghcr.io/kathanasiadis94/worker-aivyla:aivyla-global-v1 \
  --push .
```

Alternatively commit only the new files and use the independent manual GitHub Action **Build Aivyla Global (isolated)**. The workflow is visible for `workflow_dispatch` after it exists on the repository default branch; otherwise build via local Docker or promote the workflow separately.

**Do not reuse the `aivyla-5` tag** and do not update the existing production endpoint.

## Runpod Aivyla Global endpoint (not production)

1. **Active Workers = 0** during changes; don't send another job while configuring.
2. Change *Global endpoint only* to image `ghcr.io/kathanasiadis94/worker-aivyla:aivyla-global-v1`.
3. Container Disk: **start at 150 GB or more if supported**; LTX alone is 29.15 GB, plus text/audio/VAE weights and safety headroom. For workflows with >140 GB weight files, increase further. Confirm Runpod GUI limits/costs before saving.
4. Keep Global Volume `aivyla-global` mounted **exactly `/runpod-volume`**.
5. Env vars Global only:

```text
AIVYLA_GLOBAL_CACHE=1
AIVYLA_GLOBAL_LOCAL_NODES=1
AIVYLA_DYNAMIC_VRAM=0
AIVYLA_GLOBAL_CACHE_MIN_FREE_GIB=6
REFRESH_WORKER=false
```

Other existing endpoint environment variables may remain unchanged. In the **testing phase**, use Max Workers=1, Active Workers=1 for the first test, to avoid unintentionally routing the second job to a fresh worker in another region. Then compare cold vs warm latencies **on the same worker ID**. After testing return Active Workers=0 to avoid GPU cost.

## Test plan

1. Start with a small FLUX/Krea workflow and note worker ID, total duration, and `Aivyla Global CACHE MISS` logs.
2. Repeat the *same JSON* while the **same worker** remains running. Expect `CACHE HIT` / misses=0 in logs; compare timing. Note: image/video sampling speed was not GPU-benchmarked during development.
3. Test the LTX checkpoint `ltx-2.3-22b-dev-fp8.safetensors`. The cached local file should be passed to `CheckpointLoaderSimple` via ComfyUI's `is_default: true` model path; check for no `header too small` error.
4. Test one MiniMax workflow using refmods and an audio model. Dynamic or implicit model loads inside custom nodes may require additional cache mappings; report those logs rather than silently treating the workflow as production ready.
5. Return Active Workers=0, preserve old endpoint and volume until the entire suite is green.

## Boundaries / limitations

- Global object storage may require **minutes to copy a large model the first time**; this design targets **correctness and warm-worker latency**, not guaranteed fast cold starts. Warm cache is lost when worker is replaced or refreshed.
- Model caching covers file references found in `workflow[*].inputs`, for safetensors, PT, GGUF, ONNX etc. Custom-node code that constructs a path dynamically, or loads other assets from Global Volume without workflow filename input, may still access remote storage and require a second dedicated adapter.
- There is **no eviction while a worker is alive**: deleting weight files during execution can break memory maps. If the cache cannot fit another workflow, it fails with an explicit disk-space error; increase disk or recycle worker between large disjoint workflows.
- No production GPU, Runpod endpoint, or Global Volume write has been used to validate this implementation yet; complete live end-to-end testing is still required.
- Global node stage skips `.git`, dependency caches, `node_modules` and links model-like binary files >4 MiB to their original source. If a node needs such assets locally at execution, cache it separately in a subsequent iteration.

## Local validation

```bash
python3 test_global.py
```

## Rollback

For the **Aivyla Global endpoint only**, restore image tag `aivyla-5`, the old Container Disk value, and the previous Global env values. Do not change the production Aivyla endpoint. The Global Volume remains unmodified by this deployment.

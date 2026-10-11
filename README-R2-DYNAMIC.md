# Aivyla Global — dynamic R2 models & custom nodes (v4 prototype)

**Scope:** isolated test worker only. The code has not been deployed, production and Global v3 remain intact. Requires end-to-end testing in an isolated Runpod endpoint before activation.

## Goal

Upload a new LoRA, model, checkpoint or VAE into **private R2** and have the backend discover it without modifying Python model allowlists or rebuilding Docker. Allow new **custom nodes** via R2-approved archives with version pinning and checksum checks, loaded at the **next ComfyUI process startup**.

## R2 bucket layout

```text
aivyla-models/
  models/
    checkpoints/NewCheckpoint.safetensors
    diffusion_models/NewTransformer.safetensors
    loras/MyNewLoRA.safetensors
    loras/minimax_h3/MyNewLoRA.safetensors
    vae/NewVAE.safetensors
    text_encoders/NewEncoder.safetensors
    split_files/...                  # legacy-compatible (first priority)
  nodes/
    registry.json                    # explicit approved versions + activation
    packages/
      MyNewNode/v1.0.tar.gz
      MyNewNode/v1.1.tar.gz
  _aivyla/receipts/                  # optional upload SHA-256 manifests
```

**Models:** `aivyla_r2_registry.py` lists objects in `models/` at a configurable TTL. A requested model is downloaded once to the worker `/tmp` cache. The new file is discoverable even if it was not listed in the original worker image. When a matching original model exists in Global Volume it can be used as a fallback.

**Nodes:** upload a tar.gz that contains `NodeName/__init__.py` and optionally `requirements.txt`. The node becomes *available* in `nodes/registry.json`; marking `enabled: true` and starting a **new worker** installs the verified version. Nodes contain executable Python code, so simply uploading a random archive does **not** enable execution. Same name as a built-in/legacy node is rejected instead of silently overriding production logic.

**Studio UI:** the registry exposes `Registry.catalog()` and `Registry.nodes()` for a future authenticated `/api/...` endpoint. To make newly uploaded files selectable in the **Aivyla Studio app**, the app backend and its model/node selectors still need wiring. On the ComfyUI side, model workflows are fetched before prompt submission; new Python node classes appear only after a worker restart and ComfyUI loads them. Merely uploading to R2 does not by itself add new Studio UI buttons, new model-specific workflows, or UI capability definitions.

## Configurable environment — test only

| Environment | Default | Purpose |
|---|---|---|
| `AIVYLA_R2_ENABLED` | `0` | Enable lazy R2 model downloads; `0` retains v3 behavior. |
| `AIVYLA_R2_NODES_ENABLED` | `0` | Install approved nodes on next worker start. |
| `AIVYLA_R2_ENDPOINT` | *(required)* | Cloudflare account-specific S3 endpoint. |
| `AIVYLA_R2_BUCKET` | `aivyla-models` | Bucket. |
| `AWS_ACCESS_KEY_ID` | *(required)* | Store in Runpod secret/env, never git. |
| `AWS_SECRET_ACCESS_KEY` | *(required)* | Store in Runpod secret/env, never git. |
| `AIVYLA_R2_MODEL_PREFIX` | `models/` | Model key namespace. |
| `AIVYLA_R2_NODES_MANIFEST` | `nodes/registry.json` | Registry manifest key. |
| `AIVYLA_R2_CATALOG_TTL` | `60` | Model inventory cache seconds. |
| `AIVYLA_R2_DOWNLOAD_STREAMS` | `8` | Per-model parallel download threads. |
| `AIVYLA_R2_PARALLEL_MODELS` | `2` | Parallel model staging. |
| `AIVYLA_R2_FALLBACK` | `1` | Attempt Global Volume on R2 network failures. Integrity failures always fail. |
| `AIVYLA_R2_NODE_PIP` | `0` | Option to install new trusted node requirements at startup (test after compatibility review). |
| `AIVYLA_R2_NODE_MAX_ARCHIVE_GIB` | `2` | Maximum compressed node archive. |
| `AIVYLA_GLOBAL_CACHE_MIN_FREE_GIB` | `6` | Reserve disk headroom. |

**Runtime disk:** needs enough local disk for all models requested together. No eviction during execution to avoid breaking mmap. Large LTX+encoder workflows can exceed 70 GiB; provision suitable container disk *before* a live test.

## Easy uploads

Use `aivyla_r2_publish.py` on a trusted machine with `boto3` installed and read/write bucket credentials configured in environment:

```bash
# Upload a new model or LoRA. Automatic registration from its R2 key.
python3 aivyla_r2_publish.py upload-model ./my-style.safetensors --category loras
python3 aivyla_r2_publish.py upload-model ./my-model.safetensors --category diffusion_models

# Register new custom node as available, not executable.
python3 aivyla_r2_publish.py publish-node ./MyNewNode --name MyNewNode --version v1.0

# Approve/activate it for the next worker start (trusted code only).
python3 aivyla_r2_publish.py set-node MyNewNode enable

# Discover stored resources.
python3 aivyla_r2_publish.py list-models
python3 aivyla_r2_publish.py list-nodes
```

You can also upload model files straight in the Cloudflare R2 web UI, under `models/<category>/...`; these will be listed on the next catalog refresh. SHA-256 upload receipts are written by the CLI. The worker verifies receipts when available and otherwise verifies file structure/hash upon download (without an independent trusted expected hash).

## Standalone code files

- `aivyla_r2_registry.py` — dynamic S3 inventory and validated node registry.
- `aivyla_r2_stage.py` — lazy parallel download to local cache, optional Global fallback.
- `aivyla_r2_nodes.py` — approved custom-node archive installer.
- `aivyla_global_r2_handler.py` — isolated handler wrapper replacing v3 **only in test image**.
- `aivyla_global_r2_start.py` — v3 bootstrap followed by R2 node preparation.
- `aivyla_r2_publish.py` — model/LoRA/node publisher.
- `Dockerfile.global-r2` — separate experimental test image layered on v3 (not a production tag).
- `test_global_r2.py` — 15 CPU-only regression tests.

## Validation & deployment gates

Run tests before any Docker build:

```bash
python3 test_global_r2.py
python3 -m py_compile aivyla_*.py
```

**Not performed:** real Cloudflare R2 integration from inside Serverless, ComfyUI model dropdown/model loading, custom-node imports/dependency installation with actual custom nodes, Aivyla Studio authenticated catalog integration and end-to-end generation. These are required to declare the feature operational.

**No automatic build** has been enabled. Do not overwrite `ghcr.io/kathanasiadis94/worker-aivyla:aivyla-global-v3` or production `aivyla-5`. Build a NEW uniquely versioned test image only after source review and tests. Roll back by restoring the existing global-v3 image and env.


## V4.1 isolation and corrective checks (2026-10-11)

Additional regression checks cover: no Global Volume metadata lookup on a normal R2 hit,
streaming fallback if FUSE incorrectly reports `st_size=0`, integrity vs network
error handling, explicit approval for custom-node dependency installation, and
idempotent R2 startup patching. **15/15 CPU-only tests pass.** Live Serverless
behavior and Studio UI integration are still unverified.

A new LoRA or other model becomes available in the **R2 catalog** without a
new Docker build. For an existing ComfyUI workflow, the worker can download
referenced model files lazily. This does **not** automatically add a new
model-family workflow, node-specific UI control, or Studio selector. Some
ComfyUI loader dropdowns also cache available filenames and may need a
worker/ComfyUI reload to expose a new object in the native UI.

The R2 `models/` object prefix matches the 19 models uploaded in Phase 1:
`models/loras/...` and `models/split_files/...`. Additional models are not
hardcoded; use the same category convention.

### GitHub restriction and manual staging

The connected GitHub integration returned `403 Resource not accessible by
integration` for writes to `kathanasiadis94/worker-aivyla` at branch
`aivyla-global`. It **did not** modify GitHub. These files should be
reviewed/staged only into a checkout of `aivyla-global` with the included
`install_to_branch.py` script. Default is a dry run; `--apply` copies only
new isolated files into that branch's working tree (no automatic commit,
push, CI build, Docker deploy or Runpod endpoint change).

```bash
cd /path/to/worker-aivyla  # make sure you checked out aivyla-global
python3 /path/to/extracted/install_to_branch.py
python3 /path/to/extracted/install_to_branch.py --apply
python3 test_global_r2.py
```

Never overwrite `Dockerfile.global`, `aivyla_global_cache.py`, current Global
v3 files, production `aivyla-5` or its endpoint.

# Aivyla Global fast start — isolated candidate, NOT deployed

This is a conservative candidate based on `kathanasiadis94/worker-aivyla`, branch `aivyla-global`, inspected on 2026-10-11. It is **not** an acceleration guarantee and **not production-ready** until CPU regression, Docker image, actual R2 integration, ComfyUI workflows and GPU tests pass.

## What actually changes

- `aivyla_fast_requirements.py`: skip an individual node's `pip install -r` only when all its **simple** pinned/versioned requirements are already installed and meet markers/specifiers. Unknown formats (`-r`, editable, URL, extras, hashes, directives, continuations) use **exactly the old pip command**. `AIVYLA_FAST_REQUIREMENTS=0` disables the optimization.
- `aivyla_fast_start.py`: initializes the original R2/v3 handlers, preserves SageAttention wheel install and mandatory GPU test, and timestamps phases. Existing R2 staging and integrity checks remain unchanged.
- `Dockerfile.global-r2-fast`: standard test image from `aivyla-global-v3`, still stages the original node bundle on startup.
- `Dockerfile.global-r2-fast-bundled`: **optional** larger Docker image that verifies and includes the exact existing 16-node bundle during build. It skips startup extraction when the node list matches, otherwise it returns to original staging. It may cause **slower image pulls**; never assume it's the winner without benchmarks.
- The existing production Dockerfile, image tags, v3/v4 bootstraps, R2 objects and RunPod endpoints remain untouched.

## Install into a source checkout (do not use production)

```bash
# unpack the ZIP on the local machine
python3 install_faststart.py /path/to/worker-aivyla
python3 install_faststart.py /path/to/worker-aivyla --apply
cd /path/to/worker-aivyla
python3 test_fast_start.py
python3 test_global_r2.py
python3 test_global_v3.py
python3 -m py_compile aivyla_fast_start.py aivyla_fast_requirements.py
```

Installer refuses branches other than `aivyla-global`, missing baseline files or overwrites of differing candidate files. It does not commit or push.

Build an isolated image **only after checks pass**, preferably with buildx (Linux amd64) and a fresh test tag. Example (no `--push` and no deployment):

```bash
docker buildx build --platform linux/amd64 --load \
  -f Dockerfile.global-r2-fast \
  -t aivyla-global-r2-fast:local-test .
```

For the **bundled** variant, copy the existing trusted archive (currently `/runpod-volume/aivyla/custom_nodes_bundle_v1.tar.gz` on the mounted worker volume) to `<checkout>/assets/custom_nodes_bundle_v1.tar.gz` securely; ensure enough disk space. The Dockerfile verifies the pinned SHA-256 before extraction and refuses a mismatch. Then build explicitly with `-f Dockerfile.global-r2-fast-bundled`. **Do not upload this archive to GitHub.** Rebuild it only after an intentional node-version update and fresh trusted SHA-256 pin.

## RunPod pilot / rollback

Keep original Global v3/v4 images and the production endpoint intact. Create a **separate test endpoint** with one worker, no automatic scaling to multiple worker IDs during A/B comparisons, and use a new uniquely tagged image only after a successful build. Benchmark identical Krea2, Flux, MiniMax/H3, and LTX workflows twice on the **same worker**, plus a third run after a fresh worker. Compare delayTime, ComfyUI ready, bootstrap `Aivyla FASTBOOT` phases, R2 receipt/download/hash timings, load-to-GPU and actual sampler start. Avoid counting inference duration as startup.

- Fast candidate can save repeated pip startup checks. It cannot avoid **first** installation when packages aren't in the underlying image.
- Bundled candidate avoids the ~446 MB node-archive fetch/unpack on cold start only if source list matches; whether it wins depends on Docker layer pull and caching.
- Neither candidate eliminates model downloads on new workers or guarantees loaded weights in VRAM. A distinct warm-worker/model storage strategy and profiling remain necessary.
- No production endpoint update should occur until end-to-end validation and GPU visual regression testing.

Rollback: for the isolated test endpoint only, restore the previously recorded Global v3/v4 image tag and environment settings. `AIVYLA_FAST_REQUIREMENTS=0` restores the original pip behavior within the fast candidate. Avoid deleting or replacing original R2 / Global Volume data.

## Known limitations

The test suite is CPU-only. The container environment used to create it has no RunPod API session, GPU, Docker engine, original 446 MB bundle, or live R2 credentials. GitHub write operations via the available connection returned `403 Resource not accessible by integration`; no commit was made. Builds and benchmark timings have **not** been asserted as completed. The project has an existing R2 v4.2 that already implements parallel transfer and disk caching; this candidate does not alter it.

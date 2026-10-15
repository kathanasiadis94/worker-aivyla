# AvatarForever comparisons on Aivyla Global

This candidate adds an opt-in Python pipeline to the Global endpoint. The
first comparison produces BASE, OMNI1 and OMNI2 in one RunPod request, with
identical inputs, seed, prompt, dimensions and duration. Default duration is
10 seconds; the hard limit remains 40 seconds. Each variant runs in a fresh
process. BASE has no extra LoRA; OMNI1/OMNI2 use OmniNFT at 1.0/2.0.

The candidate has CPU regression coverage. Actual GPU generation and LoRA
quality must be checked on the endpoint; successful loading is not proof of
better synchronization or identity. FaceID, DubIt, distilled and MSR stacks
are not silently substituted into these experiments.

## Build and deploy

- Branch: `aivyla-global` only.
- Dockerfile: `Dockerfile.global-avatarforever`.
- Workflow: `build-aivyla-global-avatarforever.yml`. Pushes changing this
  workflow or its Dockerfile trigger CPU checks and an isolated linux/amd64
  image build on `aivyla-global`. Other source-only changes need a manual
  workflow dispatch. No workflow deploys an endpoint.
- Image tag: `ghcr.io/kathanasiadis94/worker-aivyla:aivyla-global-avatarforever-ci-<run_number>`.
- Upstream code is pinned to `de4deb6a55c9567a17dad58d67390fe089993630`.
- The isolated sibling venv shares only the installed distributions needed
  by `torch`, `torchaudio`, `torchvision`, `triton`, and pip bootstrap.
  Their dependency closure is derived from the actual base metadata and
  linked into the target, with exact constraints and bytecode writes disabled.
  The base site-packages directory and its `.pth` files are never exposed.
  ComfyUI packages, including its `diffusers`, are absent from the target.
  Other upstream dependencies are installed with their complete graph there.
  The Gemma stack is pinned to:
  `transformers==4.57.6`, `huggingface-hub==0.36.2`, `tokenizers==0.22.2`.
  These load from the sibling venv; ComfyUI retains its own versions.
  Missing packages such as accelerate are installed into the sibling venv.
  Its interpreter runs pip check, imports every packaged upstream module,
  checks real CPU attention and LoRA fusion, constructs the full 12B Gemma
  on meta through the actual upstream builder/mutator, and runs the upstream
  text encoder's forward path using a tiny CPU Gemma instance.
  No image is published unless these exact-container checks pass.
- Change only the Aivyla Global endpoint's image after a successful build.
  Record the previous image reference for rollback. Keep worker concurrency
  at 1 and configure the endpoint execution timeout to allow the comparison
  plus the initial model download (the comparison budget is 3300 seconds).
- The GPU preflight requires one visible GPU with at least 70000 MiB free.
  This is a conservative pilot gate, not a measured minimum VRAM claim.
  The selected test hardware is RTX PRO 6000 Blackwell 96 GB, one GPU.
  `AIVYLA_DYNAMIC_VRAM=0` remains explicit in this image.

Existing Global bootstrap, ComfyUI graphs and Studio services remain in use.
The production handler, image tags, endpoint and app routing are not edited.

Required existing secrets/settings:

```text
AIVYLA_R2_ENDPOINT=<existing Cloudflare R2 S3 endpoint>
AWS_ACCESS_KEY_ID=<existing secret>
AWS_SECRET_ACCESS_KEY=<existing secret>
AIVYLA_STUDIO_BASE_URL=<existing HTTPS Studio origin>
AIVYLA_OUTPUT_SHARED_SECRET=<existing secret>
```

The candidate image defaults both model buckets to `aivyla-models-v2`.
`AIVYLA_AVATARFOREVER_R2_BUCKET` controls the AvatarForever bundle explicitly.
Download parallelism is capped at two objects and twelve streams per object.

## Model staging

The bundle is staged on local worker disk under
`/tmp/aivyla-avatarforever-models`:

| R2 key/prefix | Local path under that root |
| --- | --- |
| `models/split_files/checkpoints/avatarforever-ltx-2.3-22b.safetensors` | `checkpoints/avatarforever-ltx-2.3-22b.safetensors` |
| `models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/` | `checkpoints/gemma-3-12b-it-qat-q4_0-unquantized/` |
| `models/split_files/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` | `loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` |

OmniNFT falls back to the same filename under `models/loras/` if the split
key is absent. All 16 Gemma files, including JSON/tokenizer metadata, are
required. The stager checks all requested objects before downloading and
requires disk space for all cache misses plus an 8 GiB reserve. It never
downloads weights onto the Studio server or writes to R2/network volumes.

Downloads verify size, available SHA256 metadata, and a stable final ETag.
Local cache receipts preserve the fingerprint and computed SHA256. The
Gemma index must reference the exact five shards. LoRA header mapping and
matrix shapes are checked before inference; actual fusion is audited too.
Unsupported keys or partial/zero mapping fail instead of being ignored.

ComfyUI's queue must be idle. Its model cache is unloaded through `/free`
before starting the isolated GPU process, with a check for released VRAM.
Both the capabilities check and GPU preflight wait for a valid `/queue`
response before proceeding. Startup connection failures are retried for up to
180 seconds (`AIVYLA_AVATARFOREVER_COMFY_READY_TIMEOUT_SECONDS`, at most 600).
An exited ComfyUI PID fails immediately. A busy queue is never unloaded.

## Run from the Studio server

Download `avatarforever_global_test.py` from this branch onto the Studio
server, then use `/var/www/studio-fastapi/venv/bin/python` to run it. The
client reads `/var/www/studio-fastapi/.env` and reuses the installed Studio
output reader. It never prints API keys or puts them in the job payload.

Set `AIVYLA_GLOBAL_ENDPOINT_ID` or supply `--endpoint` explicitly. The client
never falls back to the production `RUNPOD_ENDPOINT_ID`.

Check the deployed runtime without downloading models or generating video:

```bash
/var/www/studio-fastapi/venv/bin/python /var/tmp/avatarforever_global_test.py --endpoint GLOBAL_ID --check
```

Preview a comparison with the actual image and audio paths:

```bash
/var/www/studio-fastapi/venv/bin/python /var/tmp/avatarforever_global_test.py --endpoint GLOBAL_ID --image /absolute/photo.png --audio /absolute/speech.wav --seconds 10
```

Append `--execute` to submit it. `GLOBAL_ID` and the media paths above are
placeholders. Use `--variants BASE` for just the initial baseline or
`--variants OMNI1,OMNI2` for the two LoRA variants. Do not compare different
seeds/inputs as if they isolated the LoRA effect.

The client prints the job ID and a resume command immediately. Polling
errors or closing the terminal must not lead to a duplicate submission:

```bash
/var/www/studio-fastapi/venv/bin/python /var/tmp/avatarforever_global_test.py --endpoint GLOBAL_ID --resume JOB_ID
```

Outputs are saved under `/var/tmp/avatarforever-tests/<timestamp>/` as
`BASE.mp4`, `OMNI1.mp4`, `OMNI2.mp4`, `job.json` and `result.json`. Each video
keeps the supplied audio, trimmed to the requested duration. Image fitting
uses padding without cropping. Audio is padded only internally through the
last valid 8n+1 generated frame; the final soundtrack uses the original.

Results separate R2 staging, inference/model loading, upload and overall
time; they include peak GPU allocation and matched LoRA pairs. A failed
variant returns an error plus any already-delivered videos. The client
preserves those videos before reporting failure.

Review lips, facial identity, pause behavior, framing, hand artifacts and
time. Only an acceptable 10-second result advances to the same 30–40-second
sample, with pauses. Nothing routes AvatarForever into normal app jobs yet.

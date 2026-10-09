# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=0.34.0
ARG CUDA_VERSION_FOR_COMFY=12.8
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
# comfy-cli is pinned: its install/torch-index behavior decides what lands in
# the workspace venv, so an unpinned version makes builds non-reproducible.
RUN uv pip install comfy-cli==1.13.0 pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# comfy-cli installs ComfyUI into its own workspace venv (/comfyui/.venv), but
# start.sh launches ComfyUI with /opt/venv's python. That mismatch leaves the
# launch venv missing ComfyUI's runtime deps (e.g. sqlalchemy, pulled in by
# ComfyUI's asset DB), so ComfyUI crashes at startup and surfaces as the
# misleading "ComfyUI server (127.0.0.1:8188) not reachable" error. Mirror
# ComfyUI's full dependency set (core + custom nodes) into /opt/venv so the
# launch venv is complete. Root-cause fix for DR-1170.
#
# The transformers/huggingface-hub pin is part of the SAME step on purpose:
# ComfyUI declares transformers>=4.50.3 and huggingface-hub with NO upper bound,
# so a fresh install can pull transformers 5.x / huggingface-hub 1.x whose
# breaking API changes also crash ComfyUI at startup. Pinning them in the same
# RUN downgrades within one layer, so the unwanted versions aren't left behind
# bloating the image.
#
# torch is installed FIRST as the CUDA 13 (+cu130) build used in production.
# It needs host driver >= 580, so the endpoint must allow CUDA 13.0 or newer only.
RUN uv pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 \
      --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install -r /comfyui/requirements.txt \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && uv pip install -r "$r" || true; \
       done \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0"

# Build-time smoke test: actually start ComfyUI (imports the full node graph) so
# a startup-breaking dependency is caught HERE, at build time, instead of as a
# runtime "server not reachable" failure on a live worker. Runs on CPU — no GPU
# needed to exercise the import graph.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py aivyla_output_upload.py test_input.json ./RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Aivyla production image, without the FLUX demo model of the upstream image.
FROM base AS aivyla

ARG AIVYLA_COMFY_COMMIT=a8686f2b33fc540f137df50c0f0719953830a5e7

COPY aivyla/constraints.txt /etc/aivyla-constraints.txt
ENV PIP_CONSTRAINT=/etc/aivyla-constraints.txt

RUN python -m pip uninstall -y albumentationsx albumentations \
 && python -m pip install --no-cache-dir 'albumentations==2.0.8' 'albucore==0.0.24'

# Pinned ComfyUI core (same commit the network volume uses) and its requirements
RUN git clone -q --filter=blob:none https://github.com/comfyanonymous/ComfyUI /tmp/comfy-src \
 && git -C /tmp/comfy-src checkout -q "${AIVYLA_COMFY_COMMIT}" \
 && git -C /tmp/comfy-src archive --format=tar HEAD | tar -xf - -C /comfyui \
 && echo "${AIVYLA_COMFY_COMMIT}" > /comfyui/.aivyla-core-commit \
 && rm -rf /tmp/comfy-src \
 && python -m pip install --no-cache-dir -r /comfyui/requirements.txt \
 && python -m pip install --no-cache-dir --no-deps \
      comfy-aimdo==0.5.5 comfy-kitchen==0.2.35 comfyui-embedded-docs==0.5.11 \
      comfyui-frontend-package==1.52.7 comfyui-workflow-templates==0.11.62 \
      comfyui-workflow-templates-core==0.3.350 comfyui-workflow-templates-json==0.1.85 \
      comfyui-workflow-templates-media-assets-01==0.1.46

# Compiler and Python headers, needed to build insightface
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# Custom-node packages: exactly the versions production installed, in the same order
RUN python -m pip install --no-cache-dir 'ninja==1.13.0' 'wheel==0.45.1' \
 && python -m pip install --no-cache-dir --no-deps \
      cloudpickle==3.1.2 contourpy==1.4.0 cycler==0.12.1 cython==3.3.0 decorator==5.3.1 \
      easydict==1.13 flatbuffers==25.12.19 fonttools==4.66.1 imageio==2.38.1 joblib==1.6.0 \
      kiwisolver==1.5.1 lazy_loader==0.6 librosa==1.0.0 llvmlite==0.50.0 matplotlib==3.11.2 \
      ml_dtypes==0.6.0 msgpack==1.2.3 narwhals==2.26.0 numba==0.68.0 onnx==1.23.2 \
      onnxruntime==1.30.0 opencv-python==5.0.0.93 platformdirs==4.12.4 pooch==1.9.0 \
      protobuf==7.36.2 pyparsing==3.3.3 scikit-image==0.26.0 scikit-learn==1.9.1 \
      soundfile==0.14.0 soxr==1.1.0 threadpoolctl==3.7.0 tifffile==2026.9.20 \
 && python -m pip install --no-cache-dir --no-deps insightface==0.7.3 \
 && python -m pip install --no-cache-dir --no-deps opencv-python-headless==4.14.0.94 \
 && python -m pip install --no-cache-dir --no-deps \
      color-matcher==0.6.0 ddt==1.7.2 docutils==0.23 mss==10.2.0 \
 && python -m pip install --no-cache-dir --no-deps \
      diffusers==0.41.0 huggingface_hub==1.33.0 importlib_metadata==9.0.1 ninja==1.11.1.4 \
      timm==1.0.30 tokenizers==0.23.2 transformers==5.19.0 zipp==4.1.1 \
 && python -m pip install --no-cache-dir --no-deps \
      opencv-contrib-python==5.0.0.93 onnxruntime-gpu==1.30.0

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}

RUN test -f /opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvrtc.so.13 \
 && python -c "from importlib.metadata import version as v; assert v('torch') == '2.11.0+cu130', v('torch'); print('torch', v('torch'))" \
 && python -c "import cv2, onnxruntime, insightface, albumentations; print('cv2', cv2.__version__, 'ort', onnxruntime.__version__)" \
 && cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

COPY aivyla/aivyla-start.sh /aivyla-start.sh
RUN chmod +x /aivyla-start.sh

CMD ["/aivyla-start.sh"]

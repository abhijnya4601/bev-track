# GPU environment for BEVFormer-tiny training/inference. Pins are copied verbatim from
# third_party/BEVFormer/docs/install.md at the submodule's commit (66b65f3, May 2024), with two
# deviations, both marked below.
#
# Build (no GPU needed to build):  docker build -t bevtrack-gpu -f docker/Dockerfile .
# Run:  docker run --gpus all --shm-size=8g -it -v $PWD:/workspace/bev-track bevtrack-gpu
#
# GPU compatibility: CUDA 11.1 targets up to sm_86. Use a T4, V100, A10(G), A100 or RTX 30xx.
# Ada/Hopper cards (L4, RTX 40xx, H100) are newer than this toolchain and are NOT expected to work
# with these pins.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    PYTHONUNBUFFERED=1

# Ubuntu 20.04 ships Python 3.8, which is what BEVFormer's install guide uses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip git ninja-build build-essential \
        libgl1 libglib2.0-0 wget unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.8 /usr/bin/python

RUN python -m pip install --no-cache-dir --upgrade "pip<24"

RUN pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
        -f https://download.pytorch.org/whl/torch_stable.html

# Prebuilt mmcv-full wheel (the install guide's commented-out alternative); building from source takes ~30 min.
RUN pip install --no-cache-dir mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install --no-cache-dir mmdet==2.14.0 mmsegmentation==0.14.1

RUN pip install --no-cache-dir einops fvcore seaborn iopath==0.1.9 timm==0.6.13 typing-extensions==4.5.0 \
        pylint ipython==8.12 numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 \
        scikit-image==0.19.3 setuptools==59.5.0

RUN git clone https://github.com/open-mmlab/mmdetection3d.git /opt/mmdetection3d \
    && cd /opt/mmdetection3d && git checkout v0.17.1 && python setup.py install

# DEVIATION 1: the guide installs detectron2 from git HEAD, which no longer builds against
# torch 1.9. v0.6 has a prebuilt cu111/torch1.9 wheel. BEVFormer only imports it for its DD3D
# code, but the plugin's __init__ imports that unconditionally.
# Pillow 10 removed Image.LINEAR, which detectron2 0.6 uses at import time.
RUN pip install --no-cache-dir "pillow==9.5.0"
RUN pip install --no-cache-dir detectron2 \
        -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# DEVIATION 2: BEV-Track's own eval/tracking deps (pure Python; no effect on the stack above).
RUN pip install --no-cache-dir pyquaternion scipy "pytest<8.4"

WORKDIR /workspace/bev-track

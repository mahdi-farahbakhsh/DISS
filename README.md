# 🛠️ DISS: Inference-Time Search using Side Information for Diffusion-based Image Reconstruction

## Introduction
This is our paper:)

## DPS Setup
### 1) Clone Repositories

Clone the required repositories:

```bash
git clone https://github.com/DPS2022/diffusion-posterior-sampling.git integrations/dps/diffusion-posterior-sampling
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring integrations/dps/diffusion-posterior-sampling/bkse
git clone https://github.com/LeviBorodenko/motionblur integrations/dps/diffusion-posterior-sampling/motionblur
````

<br />

### 2) Setup Conda Environment

Create and activate a new Conda environment:

```bash
conda create -n DPS python=3.8
conda activate DPS
```

<br />

### 3) Install Dependencies

Install Python requirements:

```bash
pip install -r requirements/dps.txt
```

Install PyTorch with CUDA 11.3:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

<br />

### 4) Download Pretrained Models

Create the models directory and download pretrained checkpoints using `gdown`:

```bash
mkdir integrations/dps/diffusion-posterior-sampling/models
gdown 1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O integrations/dps/diffusion-posterior-sampling/models/ffhq_10m.pt
gdown 1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O integrations/dps/diffusion-posterior-sampling/models/imagenet256.pt
```

<br />

### 5) Enable Module Importability

Automatically add missing `__init__.py` files for package-style imports:

```bash
python integrations/add_inits.py
```

<br />

### 6) Apply Minimal Patch to DPS

Patch DPS with minimal required changes:

```bash
cd integrations/dps/diffusion-posterior-sampling/
git apply ../dps_modifications.patch
cd ../../..
```

<br />

### 7) Setup AdaFace

Clone AdaFace and install its dependencies:

```bash
mkdir third_party
git clone https://github.com/mk-minchul/AdaFace.git third_party/AdaFace
```

Download the pretrained AdaFace checkpoint:

```bash
mkdir third_party/AdaFace/pretrained
gdown '1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2' -O third_party/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
```

## Blind-DPS Setup

### 1) Clone Repositories

Clone the required repositories:

```bash
git clone https://github.com/BlindDPS/blind-dps.git integrations/blinddps/blind-dps
git clone https://github.com/LeviBorodenko/motionblur integrations/blinddps/blind-dps/motionblur

mkdir integrations/blinddps/blind-dps/models
gdown 1nAhgjU8C6DCkOLmWTuPIzA6PMNkNmE5Z -O integrations/blinddps/blind-dps/models/ffhq_10m.pt
gdown 11Xn8tsisCCIrv3aFyitmj55Sc13Wwb8j -O integrations/blinddps/blind-dps/models/kernel_checkpoint.pt

python integrations/add_inits.py
cd integrations/blinddps/blind-dps/
git apply ../blind_dps_modifications.patch
cd ../../..


````

# Diffusion-based Inference-time Search using Side Information (DISS)

Modular framework for inference-time search and reward-guided diffusion in image reconstruction tasks.

## 1. Clone required repositories
git clone https://github.com/DPS2022/diffusion-posterior-sampling.git integrations/dps
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring integrations/dps/diffusion-posterior-sampling/bkse
git clone https://github.com/LeviBorodenko/motionblur integrations/dps/diffusion-posterior-sampling/motionblur

## 2. Setup environment
conda create -n DPS python=3.8
conda activate DPS
pip install -r requirements/dps.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

## 3. Download pretrained models
mkdir integrations/dps/diffusion-posterior-sampling/models
gdown --id 1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O integrations/dps/diffusion-posterior-sampling/models/ffhq_10m.pt
gdown --id 1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O integrations/dps/diffusion-posterior-sampling/models/imagenet256.pt

## 4. Patch the original DPS codebase
python integrations/add_inits.py
git apply integrations/dps/dps_modifications.patch

## 5. Install AdaFace
mkdir third_party
git clone https://github.com/mk-minchul/AdaFace.git third_party/AdaFace
pip install -r requirements/adaface.txt
mkdir third_party/AdaFace/pretrained
gdown '1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2' -O third_party/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt










# Diffusion based Inference time Search using Side information (DISS)
Modular framework for inference-time search and reward-guided diffusion in image reconstruction tasks.

git clone https://github.com/DPS2022/diffusion-posterior-sampling.git integrations/dps

git clone https://github.com/VinAIResearch/blur-kernel-space-exploring integrations/dps/diffusion-posterior-sampling/bkse

git clone https://github.com/LeviBorodenko/motionblur integrations/dps/diffusion-posterior-sampling/motionblur

conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements/dps.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

mkdir integrations/dps/diffusion-posterior-sampling/models

gdown --id 1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O integrations/dps/diffusion-posterior-sampling/models/ffhq_10m.pt

gdown --id 1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O integrations/dps/diffusion-posterior-sampling/models/imagenet256.pt

python integrations/add_inits.py

git apply integrations/dps/dps_modifications.patch

in the root: 

mkdir third_party

git clone https://github.com/mk-minchul/AdaFace.git third_party/AdaFace

pip install -r requirements/adaface.txt

mkdir third_party/AdaFace/pretrained

gdown '1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2' -O third_party/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt


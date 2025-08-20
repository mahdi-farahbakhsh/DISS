from diss_modules.reward import AdaFaceReward
from typing import List
import numpy as np

import torch
import piq


def compute_lpips(x: torch.Tensor, gt: torch.Tensor, device='cuda') -> torch.Tensor:
    lpips_metric = piq.LPIPS(reduction='none').to(device)  # no net argument
    x, gt = x.to(device), gt.to(device)
    scores = lpips_metric(x, gt)  # (B,)
    return scores


def compute_psnr(x: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    x_rescaled = torch.clamp((x + 1) / 2, 0, 1)  # [-1,1] → [0,1]
    gt_rescaled = (gt + 1) / 2
    scores = piq.psnr(x_rescaled, gt_rescaled, data_range=data_range, reduction='none')
    return scores


def compute_ssim(x: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    x_rescaled = torch.clamp((x + 1) / 2, 0, 1)  # [-1,1] → [0,1]
    gt_rescaled = (gt + 1) / 2
    scores = piq.ssim(x_rescaled, gt_rescaled, data_range=data_range, reduction='none')
    return scores


def compute_face(x: torch.Tensor, gt: torch.Tensor):
    reward_network = AdaFaceReward()
    with torch.no_grad():
        gt_embed = reward_network._embeddings(gt)
        x_embed = reward_network._embeddings(x)
    return torch.norm(gt_embed - x_embed, dim=1)



import clip
import torch.nn.functional as F


def get_clip_embedding(x: torch.Tensor):
    # this will download the ViT-B/32 weights on first run
    clip_model, preprocess = clip.load("ViT-B/32", device=x.device)
    clip_model.to(x.device)

    channel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
    channel_sd = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)

    image = (x + 1) / 2

    image = F.interpolate(
        image.to(x.device),
        size=clip_model.visual.input_resolution,
        mode="bilinear",
        align_corners=False
    )
    image = (image - channel_mean) / channel_sd
    with torch.no_grad():
        embedding = clip_model.encode_image(image)  # shape [B, D]

    # normalize the embedding
    normalized_embedding = embedding / embedding.norm(dim=1, keepdim=True)
    return normalized_embedding


def compute_clip_score(x: torch.Tensor, gt: torch.Tensor):  # gt is the index in the path
    x_embedding = get_clip_embedding(x)
    gt_embedding = get_clip_embedding(gt)

    # compute cosine similarity using clip model
    clip_score = torch.sum(x_embedding * gt_embedding, dim=1).float()
    return clip_score


def get_evaluation_table_string(x: torch.Tensor, gt: torch.Tensor, metrics: List[str]) -> str:
    B = x.size(0)
    device = x.device
    fallback = 1000.0
    good_idx = (~torch.isnan(x).view(B, -1).any(dim=1) & ~torch.isnan(gt).view(B, -1).any(dim=1)).nonzero(as_tuple=True)[0]

    computed = {}
    metric_fns = {
        "lpips": compute_lpips,
        "psnr": compute_psnr,
        "ssim": compute_ssim,
        "adaface": compute_face,
        "clip": compute_clip_score
        # Add more metrics as needed
    }

    metric_names = {
        "lpips": "LPIPS",
        "psnr": "PSNR",
        "ssim": "SSIM",
        "adaface": "FaceDiff",
        "clip": "ClipScore"
    }

    # Initialize all metric tensors with fallback values
    for key in metrics:
        val = -fallback if key in ["psnr", "ssim", "adaface", "clip"] else fallback
        computed[key] = torch.full((B,), val, device=device)

    if good_idx.numel() > 0:
        x_good = x[good_idx]
        gt_good = gt[good_idx]
        for key in metrics:
            fn = metric_fns.get(key)
            if fn is not None:
                computed[key][good_idx] = fn(x_good, gt_good)

    # Convert to NumPy
    computed_np = {k: v.cpu().numpy() for k, v in computed.items()}

    # Table header
    table_str = f"{'Image':<8}" + "".join(f"{metric_names[m]:<11}" for m in metrics) + "\n"
    table_str += "-" * (8 + len(metrics) * 11) + "\n"

    # Row-wise output
    for i in range(B):
        row = f"{i:<8}" + "".join(f"{computed_np[m][i]:<11.4f}" for m in metrics) + "\n"
        table_str += row

    # Compute averages over valid samples
    valid = np.ones(B, dtype=bool)
    for v in computed_np.values():
        valid &= np.abs(v) <= 100

    if valid.any():
        row = f"{'Average':<8}" + "".join(f"{computed_np[m][valid].mean():<11.4f}" for m in metrics) + "\n"
    else:
        row = f"{'Average':<8}" + "".join(f"{np.nan:<11.4f}" for _ in metrics) + "\n"

    table_str += "-" * (8 + len(metrics) * 11) + "\n"
    table_str += row

    return table_str

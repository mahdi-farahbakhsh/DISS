import torch
import piq
from diss_modules.reward import AdaFaceReward, TextAlignmentReward, ImageReward
import re
from typing import List, Tuple
import pandas as pd
import numpy as np
import clip
import torch.nn.functional as F



__EVAL_FN__ = {}


def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            if __EVAL_FN__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __EVAL_FN__[name] = cls
        cls.name = name
        return cls

    return wrapper

def get_clip_embedding(x: torch.Tensor):
    # this will download the ViT-B/32 weights on first run
    clip_model, preprocess = clip.load("ViT-B/32", device=x.device)
    clip_model.to(x.device)

    channel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
    channel_sd = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)

    image = (x + 1) / 2

    image = F.interpolate(
        image.to(x.device),\
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


@register_eval_fn('lpips')
class LPIPS:
    def __call__(self, x: torch.Tensor, gt: torch.Tensor, device='cuda', **kwargs):
        lpips_metric = piq.LPIPS(reduction='none').to(device)  # no net argument
        x, gt = x.to(device), gt.to(device)
        scores = lpips_metric(x, gt)  # (B,)
        return scores.cpu().numpy() 


@register_eval_fn('psnr')
class PSNR:
    def __call__(self, x: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0, **kwargs) -> torch.Tensor:
        x_rescaled = torch.clamp((x + 1) / 2, 0, 1)  # [-1,1] → [0,1]
        gt_rescaled = (gt + 1) / 2
        scores = piq.psnr(x_rescaled, gt_rescaled, data_range=data_range, reduction='none')
        return scores.cpu().numpy()


@register_eval_fn('ssim')
class SSIM:
    def __call__(self, x: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0, **kwargs) -> torch.Tensor:
        x_rescaled = torch.clamp((x + 1) / 2, 0, 1) # [-1,1] → [0,1]
        gt_rescaled = (gt + 1) / 2
        scores = piq.ssim(x_rescaled, gt_rescaled, data_range=data_range, reduction='none')
        return scores.cpu().numpy()

@register_eval_fn('face_similarity')
class FaceSimilarity:
    def __call__(self, x: torch.Tensor, gt: torch.Tensor, **kwargs):
        reward_network = AdaFaceReward()
        with torch.no_grad():
            gt_embed = reward_network._embeddings(gt)
            x_embed = reward_network._embeddings(x)
        return torch.norm(gt_embed - x_embed, dim=1).cpu().numpy()


@register_eval_fn('clip_score')
class ClipScore:
    def __call__(self, x: torch.Tensor, gt: torch.Tensor, **kwargs):  # gt is the index in the path
        x_embedding = get_clip_embedding(x)
        gt_embedding = get_clip_embedding(gt)

        # compute cosine similarity using clip model
        clip_score = torch.sum(x_embedding * gt_embedding, dim=1).float()
        return clip_score.cpu().numpy()


@register_eval_fn('image_reward')
class ImageReward:
    def __call__(self, x: torch.Tensor, si_file_id: int, si_path = '../../imagenet_test_data/detailed_captions/'):  # gt is the index in the path
        reward_network = ImageReward(data_path=si_path)
        reward_network.set_side_info(si_file_id)
        with torch.no_grad():
            ir_scores = reward_network.get_reward(x)
        return ir_scores.cpu().numpy()


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


class Evaluator:
    def __init__(self, eval_fn_list):
        self.eval_fn = {}
        for eval_fn in eval_fn_list:
            self.eval_fn[eval_fn.name] = eval_fn    
    
    def __call__(self, x: torch.Tensor, gt: torch.Tensor, si_file_id: int = 0):
        metrics = {}
        for eval_fn_name, eval_fn in self.eval_fn.items():
            metrics[eval_fn_name] = float(np.mean(eval_fn(x=x, gt=gt, si_file_id=si_file_id)))
        print("metrics: ", metrics)
        return metrics
import torch
import piq
from diss_modules.reward import AdaFaceReward
import re
from typing import List, Tuple
import pandas as pd
import numpy as np


import torch
import piq


def compute_lpips(x: torch.Tensor, gt: torch.Tensor, device='cuda') -> torch.Tensor:
    lpips_metric = piq.LPIPS(reduction='none').to(device)  # no net argument
    x, gt = x.to(device), gt.to(device)
    scores = lpips_metric(x, gt)  # (B,)
    return scores


def compute_psnr(x: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    x_rescaled = (x + 1) / 2  # [-1,1] → [0,1]
    gt_rescaled = (gt + 1) / 2
    scores = piq.psnr(x_rescaled, gt_rescaled, data_range=data_range, reduction='none')
    return scores


def compute_ssim(x: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    x_rescaled = (x + 1) / 2  # [-1,1] → [0,1]
    gt_rescaled = (gt + 1) / 2
    scores = piq.ssim(x_rescaled, gt_rescaled, data_range=data_range, reduction='none')
    return scores


def compute_face(x: torch.Tensor, gt: torch.Tensor):
    reward_network = AdaFaceReward()
    with torch.no_grad():
        gt_embed = reward_network._embeddings(gt)
        x_embed = reward_network._embeddings(x)
    return torch.norm(gt_embed - x_embed, dim=1)


def get_evaluation_table_string(x: torch.Tensor, gt: torch.Tensor) -> str:
    B = x.size(0)
    device = x.device

    # 1) find which samples have any NaNs
    nan_mask = (
        torch.isnan(x).view(x.shape[0], -1).any(dim=1) |
        torch.isnan(gt).view(gt.shape[0], -1).any(dim=1)
    )  # shape (B,), True = bad

    # 2) prepare four (B,) tensors, fill with fallback
    fallback = 1000.0
    lp = torch.full((B,), fallback, device=device)
    ps = torch.full((B,), -fallback, device=device)
    ss = torch.full((B,), -fallback, device=device)
    fr = torch.full((B,), fallback, device=device)

    # 3) compute metrics only on the good indices
    good_idx = (~nan_mask).nonzero(as_tuple=True)[0]
    if good_idx.numel() > 0:
        x_good = x[good_idx]
        gt_good = gt[good_idx]

        lp_good = compute_lpips(x_good, gt_good)      # (Ngood,)
        ps_good = compute_psnr(x_good, gt_good)      # (Ngood,)
        ss_good = compute_ssim(x_good, gt_good)      # (Ngood,)
        fr_good = compute_face(x_good, gt_good)      # (Ngood,)

        lp[good_idx] = lp_good
        ps[good_idx] = ps_good
        ss[good_idx] = ss_good
        fr[good_idx] = fr_good

    # 4) move to CPU/NumPy for pretty-printing
    lp_np = lp.cpu().numpy()
    ps_np = ps.cpu().numpy()
    ss_np = ss.cpu().numpy()
    fr_np = fr.cpu().numpy()

    # 5) build your ASCII table
    table_str  = f"{'Image':<8}{'LPIPS':<11}{'PSNR':<11}{'SSIM':<11}{'FaceDiff':<11}\n"
    table_str += "-" * 48 + "\n"
    for i in range(B):
        table_str += (
            f"{i:<8}"
            f"{lp_np[i]:<11.4f}"
            f"{ps_np[i]:<11.4f}"
            f"{ss_np[i]:<11.4f}"
            f"{fr_np[i]:<11.4f}\n"
        )
    table_str += "-" * 48 + "\n"

    valid = (
            (np.abs(lp_np) <= 100) &
            (np.abs(ps_np) <= 100) &
            (np.abs(ss_np) <= 100) &
            (np.abs(fr_np) <= 100)
    )

    if valid.any():
        avg_lp = lp_np[valid].mean()
        avg_ps = ps_np[valid].mean()
        avg_ss = ss_np[valid].mean()
        avg_fr = fr_np[valid].mean()
    else:
        # no valid samples → fall back to NaN (will print “nan”)
        avg_lp = avg_ps = avg_ss = avg_fr = np.nan

    # now append the Average row using those filtered means
    table_str += (
        f"{'Average':<8}"
        f"{avg_lp:<11.4f}"
        f"{avg_ps:<11.4f}"
        f"{avg_ss:<11.4f}"
        f"{avg_fr:<11.4f}\n"
    )

    return table_str


COLS = ["Image", "LPIPS", "PSNR", "SSIM", "FaceDiff"]
FMT  = "{:<8}{:<11}{:<11}{:<11}{:<11}"
FMT_NUM = "{:<8}{:<11.4f}{:<11.4f}{:<11.4f}{:<11.4f}"

def _parse_single(table_str: str) -> pd.DataFrame:
    """Turn one ASCII table into a DataFrame, dropping its internal Average row."""
    lines = [ln for ln in table_str.strip().splitlines()
             if ln.strip() and not re.match(r"^-+$", ln)]
    data = []
    for ln in lines[1:]:                      # skip header
        tokens = ln.split()
        if tokens[0].lower() == "average":
            continue
        data.append(dict(zip(COLS,
                     [tokens[0]] + list(map(float, tokens[1:])))))
    return pd.DataFrame(data)


def _format(df: pd.DataFrame, threshold: float = 100.0) -> str:
    """Return a string in the original layout, appending an Average row
       but ignoring any metric whose absolute value exceeds `threshold`."""
    parts = [FMT.format(*COLS), "-" * 48]
    # 1) print all the data rows as before
    for _, r in df.iterrows():
        parts.append(
            FMT_NUM.format(
                str(r["Image"]),
                r.LPIPS, r.PSNR, r.SSIM, r.FaceDiff
            )
        )
    parts.append("-" * 48)

    # 2) build an average, masking out any extreme fallback values
    metrics = df[["LPIPS", "PSNR", "SSIM", "FaceDiff"]]
    # create a mask of “valid” entries (|value| <= threshold)
    valid_mask = metrics.abs() <= threshold
    # turn outliers into NaN so that `.mean()` will skip them
    masked = metrics.where(valid_mask, np.nan)
    avg = masked.mean()

    parts.append(
        FMT_NUM.format(
            "Average",
            avg["LPIPS"],
            avg["PSNR"],
            avg["SSIM"],
            avg["FaceDiff"]
        )
    )
    return "\n".join(parts)


def build_tables(tables: List[str],
                 num_particles: int,
                 total_particles: int
                 ) -> Tuple[str, str, str]:
    """
    Parameters
    ----------
    tables : list[str]
        Each string is an ASCII table exactly like the example.
    num_particles : int
        Block size used to choose the min‑FaceDiff row inside the joined table.
    total_particles : int
        Total particles per *image*. Table 3 groups rows in Table 2 by
        (total_particles // num_particles).

    Returns
    -------
    joined_str, best_particles_str, per_image_str : str, str, str
        The three tables, already formatted.
    """
    # --- T1 : joined --------------------------------------------------------
    joined = pd.concat([_parse_single(t) for t in tables], ignore_index=True)
    t1 = _format(joined)

    # --- T2 : best‑of‑particles -------------------------------------------
    best_rows = []
    for start in range(0, len(joined), num_particles):
        chunk = joined.iloc[start:start+num_particles]
        if not chunk.empty:
            best_rows.append(chunk.loc[chunk.FaceDiff.idxmin()])
    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    t2 = _format(best_df)

    # --- T3 : per‑image averages ------------------------------------------
    group_size = total_particles // num_particles
    if group_size == 0:
        raise ValueError("total_particles must be ≥ num_particles.")
    per_img = []
    for start in range(0, len(best_df), group_size):
        grp = best_df.iloc[start:start+group_size]
        if not grp.empty:
            mean_vals = grp[["LPIPS","PSNR","SSIM","FaceDiff"]].mean()
            per_img.append({"Image": start//group_size,
                            **mean_vals.to_dict()})
    per_df = pd.DataFrame(per_img)
    t3 = _format(per_df)

    return t1, t2, t3

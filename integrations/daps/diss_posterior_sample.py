import json
import yaml
import torch
from torchvision.utils import save_image
from forward_operator import get_operator
from data import get_dataset
from sampler import get_sampler, Trajectory
from model import get_model
from eval import get_eval_fn, Evaluator
from diss_modules.reward import get_reward_method, MeasurementReward
from diss_modules.search import get_search_method
from diss_modules.eval import get_evaluation_table_string
from torch.nn.functional import interpolate
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb
import setproctitle
from PIL import Image
import numpy as np
import imageio
import os


def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry


def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)


def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils


def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)


def save_mp4_video(gt, y, x0hat_traj, x0y_traj, xt_traj, output_path, fps=24, sec=5, space=4):
    """
        stack and save trajectory as mp4 video
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    ix, iy = x0hat_traj.shape[-2:]
    reindex = np.linspace(0, len(xt_traj) - 1, sec * fps).astype(int)
    np_x0hat_traj = tensor_to_numpy(x0hat_traj[reindex])
    np_x0y_traj = tensor_to_numpy(x0y_traj[reindex])
    np_xt_traj = tensor_to_numpy(xt_traj[reindex])
    np_y = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]
    for x0hat, x0y, xt in zip(np_x0hat_traj, np_x0y_traj, np_xt_traj):
        canvas = np.ones((ix, 5 * iy + 4 * space, 3), dtype=np.uint8) * 255
        cx = cy = 0
        canvas[cx:cx + ix, cy:cy + iy] = np_y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = np_gt

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0hat

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = xt
        writer.append_data(canvas)
    writer.close()


def save_batch_reconstructions(root: Path, s: int, cur_samples: torch.Tensor, cur_y: torch.Tensor):
    """
    root: Path to args.save_dir / args.name
    s: batch start index (0, batch_size, 2*batch_size, …)
    cur_samples: Tensor[B,3,H,W]
    cur_y:        Tensor[B,3,H,W]  (all entries identical)
    """
    # 1) make sure dirs exist
    recon_dir = safe_dir(root / 'reconstruction')
    y_dir     = safe_dir(root / 'y')

    # 2) save each reconstructed sample
    #    restore to [0,1] range with the same scheme you use elsewhere:
    for idx, img in enumerate(cur_samples):
        # img * 0.5 + 0.5 puts it back into [0,1]
        out_path = recon_dir / f'{s:05d}_{idx:03d}.png'
        save_image(img * 0.5 + 0.5, fp=str(out_path))

    # 3) save one measurement image
    meas_path = y_dir / f'{s:05d}.png'
    save_image(cur_y[0] * 0.5 + 0.5, fp=str(meas_path))

def log_results(args, sde_trajs, results, images, y, full_samples, table_markdown, total_number):
    # log hyperparameters and configurations
    full_samples = full_samples.flatten(0, 1)
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)
    with open(str(root / 'config.yaml'), 'w') as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file, default_flow_style=False, allow_unicode=True)

    # log grid results
    resized_y = resize(y, images, args.task[args.task_group].operator.name)
    stack = torch.cat([images, resized_y, full_samples])
    save_image(stack * 0.5 + 0.5, fp=str(root / 'grid_results.png'), nrow=total_number)

    # log individual sample instances
    if args.save_samples:
        pil_image_list = tensor_to_pils(full_samples)
        image_dir = safe_dir(root / 'samples')
        cnt = 0
        for run in range(args.num_runs):
            for idx in range(total_number):
                image_path = image_dir / '{:05d}_run{:04d}.png'.format(idx, run)
                pil_image_list[cnt].save(str(image_path))
                cnt += 1

    # log sampling trajectory and mp4 video
    if args.save_traj:
        traj_dir = safe_dir(root / 'trajectory')
        print()
        print('save trajectories to mp4 videos...')
        for run, sde_traj in enumerate(sde_trajs):
            if args.save_traj_raw_data:
                # might be SUPER LARGE
                traj_raw_data = safe_dir(traj_dir / 'raw')
                torch.save(sde_traj, str(traj_raw_data / 'trajectory_run{:04d}.pth'.format(run)))
            # save mp4 video for trajectories
            x0hat_traj = sde_traj.tensor_data['x0hat']
            x0y_traj = sde_traj.tensor_data['x0y']
            xt_traj = sde_traj.tensor_data['xt']
            for idx in range(total_number):
                video_path = str(traj_dir / '{:05d}_run{:04d}.mp4'.format(idx, run))
                save_mp4_video(images[idx], resized_y[idx], x0hat_traj[:, idx], x0y_traj[:, idx], xt_traj[:, idx], video_path)

    # log the evaluation metrics
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(table_markdown)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)


def sample_in_batch(sampler, model, x_start, operator, y, evaluator, verbose, record, gt, search_rewards,
                    gradient_rewards, search, batch_size, save_root: Path = None):
    """
        posterior sampling in batch
    """
    samples = []
    trajs = []
    all_tables = []
    for s in range(0, len(x_start), batch_size):
        # update evaluator to correct batch index
        for reward in search_rewards + gradient_rewards:
            reward.set_side_info(s // batch_size)
        cur_x_start = x_start[s:s + batch_size]
        cur_y = y[s:s + batch_size]
        cur_gt = gt[s: s + batch_size]
        cur_samples = sampler.sample(model, cur_x_start, operator, cur_y, search_rewards, gradient_rewards, search, evaluator,
                                     verbose=verbose, record=record, gt=cur_gt)

        samples.append(cur_samples)
        if record:
            trajs.append(sampler.trajectory.compile())

        # —— NEW: save these batch results if save_root was provided
        if save_root is not None:
            save_batch_reconstructions(save_root, s//batch_size, cur_samples, cur_y)

        table = get_evaluation_table_string(cur_samples, cur_gt, metrics=['psnr', 'lpips', 'ssim', 'adaface'])
        all_tables.append(table)
        print(table)

    if record:
        trajs = Trajectory.merge(trajs)

    return torch.cat(samples, dim=0), trajs, all_tables


@hydra.main(version_base='1.3', config_path='diss_configs', config_name='default.yaml')
def main(args):
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.gpu))

    setproctitle.setproctitle(args.name)
    print(args)

    # get rewards
    rewards_cfg = args.reward.get('rewards', [])
    gradient_rewards = [
        get_reward_method(cfg['name'], **{k: v for k, v in cfg.items() if k not in ['name', 'steering']})
        for cfg in rewards_cfg if 'gradient' in cfg.get('steering', [])
    ]

    search_rewards = [
        get_reward_method(cfg['name'], **{k: v for k, v in cfg.items() if k not in ['name', 'steering']})
        for cfg in rewards_cfg if 'search' in cfg.get('steering', [])
    ]

    # get data
    num_particles = args.reward['num_particles']
    data = get_dataset(**args.data)
    total_number = len(data)  # number of all images
    img_1p = data.get_data(total_number, 0)  # shape=(total_number * particles, 3, 256, 256) - all the images
    images = torch.stack([img_1p[i] for i in range(img_1p.shape[0]) for _ in range(num_particles)], dim=0)

    # get operator & measurement
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y_1p = operator.measure(img_1p)
    y = torch.stack([y_1p[i] for i in range(y_1p.shape[0]) for _ in range(num_particles)], dim=0)
    y = y + 0 * torch.randn_like(y)
    for rew in gradient_rewards + search_rewards:
        if isinstance(rew, MeasurementReward):
            rew.set_operator(operator)

    # get sampler
    sampler = get_sampler(**args.sampler, mcmc_sampler_config=task_group.mcmc_sampler_config)

    # get model
    model = get_model(**args.model)

    # get search algorithm
    search = get_search_method(num_particles=num_particles, **args.reward['search_algorithm']) if search_rewards else None

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # +++++++++++++++++++++++++++++++++++
    # main sampling process
    full_samples = []
    full_trajs = []
    for r in range(args.num_runs):
        print(f'Run: {r}')
        x_start = sampler.get_start(images.shape[0], model)
        samples, trajs, all_tables = sample_in_batch(
            sampler, model, x_start, operator, y, evaluator, verbose=True, record=args.save_traj, gt=images,
            search_rewards=search_rewards, gradient_rewards=gradient_rewards, search=search, batch_size=num_particles,
            save_root=safe_dir(safe_dir(Path(args.save_dir)) / args.name)
        )
        #samples, trajs = images, None
        full_samples.append(samples)
        full_trajs.append(trajs)
    full_samples = torch.stack(full_samples, dim=0)

    print(f'finish {args.name}!')


if __name__ == '__main__':
    main()

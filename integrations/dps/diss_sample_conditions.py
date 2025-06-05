from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

#from menpo.feature import gradient

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from diss_modules.reward import get_reward_method
from diss_modules.search import get_search_method
from diss_modules.eval import get_evaluation_table_string, build_tables


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    # logger
    logger = get_logger()

    # Device setting
    # device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'  # changed here to work with mps
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    # "learn_sigma must be the same for model and diffusion configuartion."

    # Load model
    # Save current working directory
    original_dir = os.getcwd()
    os.chdir('diffusion-posterior-sampling')

    # Load model (this will now find the relative model path correctly)
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Return to original directory
    os.chdir(original_dir)

    # get rewards:
    rewards_cfg = task_config['rewards']
    gradient_rewards = [
        get_reward_method(cfg['name'], **{k: v for k, v in cfg.items() if k not in ['name', 'steering']})
        for cfg in rewards_cfg if 'gradient' in cfg.get('steering', [])
    ]

    search_rewards = [
        get_reward_method(cfg['name'], **{k: v for k, v in cfg.items() if k not in ['name', 'steering']})
        for cfg in rewards_cfg if 'search' in cfg.get('steering', [])
    ]

    # get search algorithm:
    num_particles = task_config['num_particles']
    MAX_BATCH_SIZE = 8
    batch_size = num_particles if num_particles <= MAX_BATCH_SIZE else MAX_BATCH_SIZE
    search = get_search_method(num_particles=batch_size, **task_config['search_algorithm']) if search_rewards else None

    # set the random seed
    seed = 37
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'], args.path)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
            **measure_config['mask_opt']
        )

    all_tables = []
    num_runs = num_particles // batch_size

    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        for reward in search_rewards + gradient_rewards:
            reward.set_side_info(i)

        for run in range(num_runs):
            # Exception) In case of inpainging,
            if measure_config['operator']['name'] == 'inpainting':
                mask = mask_gen(ref_img)
                mask = mask[:, 0, :, :].unsqueeze(dim=0)
                measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

                # Forward measurement model (Ax + n)
                y = operator.forward(ref_img, mask=mask)
                y_n = noiser(y)

            else:
                # Forward measurement model (Ax + n)
                y = operator.forward(ref_img)
                y_n = noiser(y)

            y_n = y_n.repeat(batch_size, 1, 1, 1)

            # Sampling
            x_start = torch.randn((batch_size, *ref_img.shape[1:]), device=device).requires_grad_()
            sample = sample_fn(
                x_start=x_start,
                measurement=y_n,
                record=False,
                save_root=out_path,
                gradient_rewards=gradient_rewards,
                search_rewards=search_rewards,
                search=search
            )

            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n[0].unsqueeze(0)))
            plt.imsave(os.path.join(out_path, 'label', 'img_' + fname), clear_color(ref_img))
            for particle in range(batch_size):
                plt.imsave(
                    os.path.join(out_path, 'recon', 'img_' + fname + '_' + str(particle + run * batch_size) + '.png'),
                    clear_color(sample[particle].unsqueeze(0))
                )

            logger.info('')
            table = get_evaluation_table_string(sample, ref_img.repeat(batch_size, 1, 1, 1))
            all_tables.append(table)
            print(table)

    print()
    for idx, table in enumerate(all_tables):
        print(f'results for image {idx // num_runs} and run {idx - (idx // num_runs) * num_runs}')
        print(table)
        print()

    t1, t2, t3 = build_tables(all_tables, search.max_group, num_particles)
    print(t1, '\n\n', t2, '\n\n', t3)

    print()
    print('saved in ', args.path)


if __name__ == '__main__':
    main()

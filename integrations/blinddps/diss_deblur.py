from functools import partial
import os
import argparse
import yaml
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator, get_noise
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from motionblur.motionblur import Kernel
from util.img_utils import Blurkernel, clear_color
from util.logger import get_logger
from diss_modules.reward import get_reward_method
from diss_modules.search import get_search_method
from diss_modules.eval import get_evaluation_table_string, build_tables


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch current GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs (if using multi-GPU)

    # Ensure deterministic behavior for CuDNN backend
    # these two lines will make the code significantly slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # should be false for determinism

    # For extra safety, set environment variable (optional but recommended)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    # set seed for reproduce
    seed = 120
    set_seed(seed)

    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/motion_deblur_config.yaml')
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=0.1)
    parser.add_argument('--reg_ord', type=int, default=0, choices=[0, 1])
    # Save Directory
    parser.add_argument('--path', type=str, default='')

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
    img_model_config = load_yaml(args.img_model_config)
    kernel_model_config = load_yaml(args.kernel_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

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

    # Kernel configs to namespace save space
    args.kernel = task_config["kernel"]
    args.kernel_size = task_config["kernel_size"]
    args.intensity = task_config["intensity"]


    # Save current working directory
    original_dir = os.getcwd()
    os.chdir('blind-dps')
    # Load model
    img_model = create_model(**img_model_config)
    img_model = img_model.to(device)
    img_model.eval()
    kernel_model = create_model(**kernel_model_config)
    kernel_model = kernel_model.to(device)
    kernel_model.eval()
    model = {'img': img_model, 'kernel': kernel_model}
    # Return to original directory
    os.chdir(original_dir)

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    measurement_cond_fn = cond_method.conditioning

    # Add regularization
    # Not to use regularization, set reg_scale = 0 or remove this part.
    regularization = {'kernel': (args.reg_ord, args.reg_scale)}
    measurement_cond_fn = partial(measurement_cond_fn, regularization=regularization)
    if args.reg_scale == 0.0:
        logger.info(f"Got kernel regularization scale 0.0, skip calculating regularization term.")
    else:
        logger.info(f"Kernel regularization : L{args.reg_ord}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'], args.path)
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    all_tables = []
    num_runs = num_particles // batch_size

    # Do Inference
    for i, ref_img in enumerate(loader):
        for reward in search_rewards + gradient_rewards:
            reward.set_side_info(i)

        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        for run in range(num_runs):
            try:
                if args.kernel == 'motion':
                    kernel = Kernel(size=(args.kernel_size, args.kernel_size), intensity=args.intensity).kernelMatrix
                    kernel = torch.from_numpy(kernel).type(torch.float32)
                    kernel = kernel.to(device).view(1, 1, args.kernel_size, args.kernel_size)
                elif args.kernel == 'gaussian':
                    conv = Blurkernel('gaussian', kernel_size=args.kernel_size, device=device)
                    kernel = conv.get_kernel().type(torch.float32)
                    kernel = kernel.to(device).view(1, 1, args.kernel_size, args.kernel_size)

                # Forward measurement model (Ax + n)
                y = operator.forward(ref_img, kernel)
                y_n = noiser(y)

                y_n = y_n.repeat(batch_size, 1, 1, 1)
                kernel = kernel.repeat(batch_size, 1, 1, 1)

                # Set initial sample
                # !All values will be given to operator.forward(). Please be aware it.
                x_start = {'img': torch.randn(y_n.shape, device=device).requires_grad_(),
                           'kernel': torch.randn(kernel.shape, device=device).requires_grad_()}

                # !prior check: keys of model (line 74) must be the same as those of x_start to use diffusion prior.
                for k in x_start:
                    if k in model.keys():
                        logger.info(f"{k} will use diffusion prior")
                    else:
                        logger.info(f"{k} will use uniform prior.")

                # sample
                sample = sample_fn(
                    x_start=x_start,
                    measurement=y_n,
                    record=False,
                    save_root=out_path,
                    gradient_rewards=gradient_rewards,
                    search_rewards=search_rewards,
                    search=search
                )
                print('min is: ', torch.min(sample['img']))
                print('max is: ', torch.max(sample['img']))
                # sample = {}
                # sample['img'] = ref_img.repeat(batch_size, 1, 1, 1)
                # sample['kernel'] = kernel

                plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n[0].unsqueeze(0)))
                plt.imsave(os.path.join(out_path, 'label', 'ker_' + fname), clear_color(kernel[0].unsqueeze(0)))
                plt.imsave(os.path.join(out_path, 'label', 'img_' + fname), clear_color(ref_img))
                for particle in range(batch_size):
                    plt.imsave(os.path.join(out_path, 'recon',
                                            'img_' + fname + '_' + str(particle + run * batch_size) + '.png'),
                               clear_color(sample['img'][particle].unsqueeze(0)))

                    plt.imsave(os.path.join(out_path, 'recon',
                                            'ker_' + fname + '_' + str(particle + run * batch_size) + '.png'),
                               clear_color(sample['kernel'][0].unsqueeze(0)))

                logger.info('')
                table = get_evaluation_table_string(sample['img'], ref_img.repeat(batch_size, 1, 1, 1))
                all_tables.append(table)
                print(table)
            except Exception as e:
                # write out the error
                print(f"Error occurred: {e}")

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

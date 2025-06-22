from functools import partial
import os
import argparse
import yaml
import sys

# Add parent directory to Python path to import diss_modules
sys.path.append('../../')

from diss_modules.reward import get_reward_method
from diss_modules.search import get_search_method
from diss_modules.eval import get_evaluation_table_string, build_tables

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model

from guided_diffusion.diss_guided_diffusion import create_sampler  # changed to search guided gaussian diffusion


from dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import torchvision

import numpy as np
import random
from PIL import Image
import glob
from pathlib import Path

def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str, default='./diss_configs/diss_super_resolution_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=100)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--method', type=str, default='mpgd_wo_proj')
    parser.add_argument('--save_dir', type=str, default='./outputs/ffhq/')
    parser.add_argument('--eval_fn_list', type=str, nargs='+', default=['psnr', 'ssim', 'lpips', 'facenet_l2', 'adaface_l2'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_images', type=int, default=1)

    # parser.add_argument('--search_algo_config', type=str, default='./diss_configs/search_resample.yaml')
    # parser.add_argument('--reward_eval_config', type=str, default='./diss_configs/rewards_adaface_measurement.yaml')
    # parser.add_argument('--ref_faces_path', type=str, default='../../data/additional_images/')

    # # additional args for lookahead 
    # parser.add_argument('--best_of_n', action='store_true', help='Pick out the best of n samples')
    # parser.add_argument('--num_lookahead_steps', type=int, default=1)
    # parser.add_argument('--conditional_lookahead', action='store_true')
    # parser.add_argument('--perform_lookahead', action='store_true')
    # parser.add_argument('--n_images', type=int, default=1)
    # parser.add_argument('--temp', type=float, default=1.0)
    # parser.add_argument('--num_particles', type=int, default=1)
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--resample_rate', type=int, default=4)
    # parser.add_argument('--record_inside_la', action='store_true', help='Record the lookahead samples')
    # parser.add_argument('--jump_size', type=int, default=1)
    # parser.add_argument('--jump_la', action='store_true', help='Use jump lookahead')
    # parser.add_argument('--end_resample', type=float, default=0.9)
    # parser.add_argument('--gradient_scale', type=float, default=1.0)

    args = parser.parse_args()

    # # Load model
    # # Save current working directory
    # original_dir = os.getcwd()
    # os.chdir('DISS/integrations/mpgd')
   
    # logger
    logger = get_logger()

    # Set random seed
    seed_everything(args.seed)
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    # search_algo_config = load_yaml(args.search_algo_config)    
    
    if args.timestep < 1000:
        diffusion_config["timestep_respacing"] = f"ddim{args.timestep}"
        diffusion_config["rescale_timesteps"] = True
    else:
        diffusion_config["timestep_respacing"] = f"1000"
        diffusion_config["rescale_timesteps"] = False
    
    diffusion_config["eta"] = args.eta
    task_config["conditioning"]["method"] = args.method
    task_config["conditioning"]["params"]["scale"] = args.scale
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    # cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/celeba256/model.ckpt", **cond_config['params']) # in the paper we used this checkpoint
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, resume = "../nonlinear/SD_style/models/ldm/ffhq256/model.ckpt", **cond_config['params']) # you can probably also use this checkpoint, but you probably want to tune the hyper-parameter a bit
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)


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

    print(f'batch size: {batch_size}')

    # print(f"Search algorithm: {search_algo_config['name']}")

    # reward_configs = load_yaml(args.reward_eval_config)
    # reward_eval = {}
    # from reward_eval import get_reward_eval
    # for reward_config in reward_configs:
    #     reward = get_reward_eval(**reward_config)
    #     reward_eval[reward_config['name']] = reward
    #     reward.gradient_scale = args.gradient_scale  # set the gradient scale for the reward
    #     print('reward grad scale:', reward.gradient_scale)

    # search_algo_config['num_particles'] = args.num_particles  # change the number of particles
    # search_algo_config['init_temp'] = args.temp  # change the init temp
    # search_algo_config['resample_rate'] = args.resample_rate  # change the resample rate
    # search_algo_config['num_steps'] = args.timestep  # change the number of lookahead steps
    # search_algo = get_search_algo(**search_algo_config)  # fixed
    # num_particles = search_algo_config['num_particles']

    # if args.best_of_n:
    #     search_algo = None  # disable search algorithm 
   
    # Working directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = f"{timestamp}_{diffusion_config['timestep_respacing']}_eta{args.eta}_scale{args.scale}"

    # if args.best_of_n:
    #     dir_path += f"_best_of_n_{num_particles}"
    # else:
    #     dir_path += f"_{search_algo_config['name']}_n_{num_particles}"


    # if search_algo is not None:
    #     dir_path += f"_temp_{search_algo_config['init_temp']}"
    #     dir_path += f"_resample_rate_{search_algo_config['resample_rate']}"

    # for reward_name, reward in reward_eval.items():
        
    #     dir_path += f"_{reward_name}"
    #     if reward.gradient:
    #         dir_path += f"_grad_{reward.gradient_scale}"
    #     if search_algo is not None:
    #         dir_path += f"_search"


    task_name = measure_config['operator']['name']

    if task_name == 'super_resolution':
        task_name = f"{task_name}_x{measure_config['operator']['scale_factor']}"
    out_path = os.path.join(args.save_dir, task_name, task_config['conditioning']['method'], dir_path)
    
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'guid', 'metrics_md', 'metrics_json']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # # Exception) In case of inpainting, we need to generate a mask 
    # if measure_config['operator']['name'] == 'inpainting':
    #     mask_gen = mask_generator(
    #        **measure_config['mask_opt']
    #     )

    from diss_modules.eval import get_evaluation_table_string, build_tables

    # # get evaluator
    # eval_fn_list = []
    # for eval_fn_name in args.eval_fn_list:
    #     eval_fn_list.append(get_eval_fn(eval_fn_name))
    # evaluator = Evaluator(eval_fn_list)

    img_size = 256
    # transform = transforms.Compose([
    #         transforms.Resize((img_size, img_size)),
    #         transforms.ToTensor(),
    #         transforms.Lambda(lambda t: (t * 2) - 1)
    #     ])

    # from glob import glob

    # ref_faces = sorted(glob(os.path.join(args.ref_faces_path + '/*.png')))

    # extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    # ref_faces = [file for ext in extensions for file in Path().rglob(ext)]

    n_images = args.n_images

    images = []
    samples = []
    best_samples = []

    # print(f'batch size: {args.batch_size}')
    print(f'num particles: {num_particles}')
    print(f'n_images: {n_images}')


       # # log metrics
    # n_uniq_samples = len(samples) // num_particles

    # add all the configs to the markdown table
    markdown_table = f'arguments: \n \n'
    for arg, value in vars(args).items():
        markdown_table += f'- **{arg}**: {value} \n '

    # # print reward eval configs
    # markdown_table += f' \n reward eval configs: \n \n'
    # for reward_config in reward_configs:
    #     markdown_table += f'- **{reward_config["name"]}**: \n'
    #     for key, value in reward_config.items():
    #         if key != 'name':
    #             markdown_table += f'  - {key}: {value} \n'
    # markdown_table += f' \n \n'
    # # print search algo configs
    # markdown_table += f' \n search algo configs: \n \n'
    # for key, value in search_algo_config.items():
    #     markdown_table += f'- **{key}**: {value} \n'
    # markdown_table += f' \n \n'

    all_tables = []
    num_runs = 1
        
    # Do Inference
    for i, ref_img in enumerate(loader):

        # set the side info for the rewards
        for reward in search_rewards + gradient_rewards:
            reward.set_side_info(i)

        # ref_face_img = Image.open(ref_faces[i]).convert('RGB')

        # ref_face_img = transform(ref_face_img)
        # ref_face_img = ref_face_img.to(device)
        # ref_face_img = ref_face_img.unsqueeze(0)

        # print(f"ref_face_img: {ref_face_img}")
        
        # if i >= n_images:
        #     break
        # logger.info(f"Inference for image {i}")
        fname = f'{i:03}'

        ref_img = ref_img.to(device)

        print(f'ref_img shape: {ref_img.shape}')
        # print(f'ref_face_img shape: {ref_face_img.shape}')

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        # Sampling
        x_start = torch.randn((batch_size, 3, img_size, img_size), device=device).requires_grad_()
        print(f"x_start shape: {x_start.shape}")

        plt.imsave(os.path.join(out_path, 'input', f'{i:03}_input.png'), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', f'{i:03}_label.png'), clear_color(ref_img))
        # plt.imsave(os.path.join(out_path, 'guid', f'{i:03}_guid.png'), clear_color(ref_face_img))


        sample = sample_fn(x_start=x_start,
                           measurement=y_n, 
                           record=False, 
                           save_root=out_path, 
                           cond_scale=args.scale,
                           gradient_rewards=gradient_rewards,
                           search_rewards=search_rewards,
                           search=search
                        )

        for particle in range(batch_size):
            plt.imsave(
                os.path.join(out_path, 'recon', 'img_' + fname + '_' + str(particle + batch_size) + '.png'),
                clear_color(sample[particle].unsqueeze(0))
            )
            print(f'saved {particle}th particle')


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
    print('saved in ', out_path)


    # images = torch.cat(images, dim=0)  # n_images, 3, img_size, img_size
    # samples = torch.cat(samples, dim=0)  # n_images, num_groups, num_particles, 3, img_size, img_size
    # best_samples = torch.cat(best_samples, dim=0)  # n_images, num_groups, 3, img_size, img_size

    # print(f"images shape: {images.shape}")
    # print(f"samples shape: {samples.shape}")

    # print(f"best_samples before reshape: {best_samples.shape}")
    

    # # swap the first two dimensions of best_samples
    # best_samples = best_samples.permute(1, 0, 2, 3, 4)  # num_groups, n_images, 3, img_size, img_size
    # gt_images = images.expand_as(best_samples)  # num_groups, n_images, 3, img_size, img_size

    # print(f"best_samples after reshape: {best_samples.shape}")
    # print(f"gt_images shape: {gt_images.shape}")

    # best_samples = best_samples.reshape(-1, 3, img_size, img_size)  # num_groups * n_images, 3, img_size, img_size
    # gt_images = gt_images.reshape(-1, 3, img_size, img_size)  # num_groups * n_images, 3, img_size, img_size

    # print(f"best_samples after final reshape: {best_samples.shape}")
    # print(f"gt_images final shape: {gt_images.shape}")

 
    # # for n in range(num_particles):
    # #     idxs = np.arange(n_uniq_samples) * num_particles + n
    # #     print('idxs:', idxs)
    # #     results = evaluator.report(images, y, samples[idxs])
    # #     markdown_text = evaluator.display(results)
    # #     markdown_table += '\n' + markdown_text

    # best_results = evaluator.report(gt_images, y, best_samples)  # save only the best samples and the best images
    # best_markdown_text = evaluator.display(best_results)    
    # markdown_table += '\n \n \n best results \n' + best_markdown_text

    # print(markdown_table)

    

if __name__ == '__main__':
    main()

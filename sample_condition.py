import os
from functools import partial
import argparse

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import LD_SMC
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator
from util.tools import set_seed, get_device
from util.logger import get_logger
from ldm_inverse.measurements import get_noise, get_operator
from model_loader import load_model_from_config, load_yaml


def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model


parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/ldm/ffhq/model.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/inpainting_config.yaml", type=str)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--ddim_steps', default=1000, type=int)
parser.add_argument('--ddim_eta', default=1.0, type=float)
parser.add_argument('--ddim_scale', default=1.0, type=float)

parser.add_argument('--kappa1', default=1.0, type=float, help='first gradient scaler')
parser.add_argument('--kappa2', default=2.5, type=float, help='second gradient scaler')
parser.add_argument('--rho', default=0.75, type=float, help='balance between two terms')
parser.add_argument("--s", type=int, default=333, help="switch step between elements in the proposal")
parser.add_argument("--num_particles", type=int, default=1, help="number of particles")
parser.add_argument("--num_gibbs_iters", type=int, default=1, help="number of Gibbs iterations")

# experiment
parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
parser.add_argument("--seed", type=int, default=42, help="seed value")

args = parser.parse_args()

# Load configurations
task_config = load_yaml(args.task_config)
data_config = OmegaConf.load(args.ldm_config)

set_seed(args.seed)
device = get_device(cuda=int(args.gpu) >= 0, gpus=args.gpu)

logger = get_logger()

dataset_name = data_config.data.name
task_name = task_config['measurement']['operator']['name']

if task_name == 'inpainting':
    task_name += f"_{task_config['measurement']['mask_opt']['mask_type']}"

# Loading model
model = get_model(args)
sampler = LD_SMC(model, kappa1=args.kappa1, kappa2=args.kappa2, rho=args.rho,
                 num_particles=args.num_particles, s=args.s,
                 num_gibbs_iters=args.num_gibbs_iters, logger=logger)

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Instantiating sampler
sample_fn = partial(sampler.ld_smc_sampler, operator_fn=operator.forward, task_name=task_name,
                    S=args.ddim_steps, cond_method=task_config['conditioning']['main_sampler'],
                    conditioning=None, batch_size=1,
                    shape=[3, 64, 64],  # Dimension of latent space
                    verbose=False,
                    unconditional_guidance_scale=args.ddim_scale,
                    unconditional_conditioning=None,
                    eta=args.ddim_eta)

# Working directory
out_path = os.path.join(args.save_dir)
os.makedirs(out_path, exist_ok=True)
for img_dir in ['recon', 'label']:
    os.makedirs(os.path.join(out_path, task_name, img_dir), exist_ok=True)

# Prepare dataloader
imsize = int(data_config.data.params.validation.params.size)
data_config = task_config['data']
transform = transforms.Compose([transforms.Resize(imsize),
                                transforms.CenterCrop(imsize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

# Exception) In case of inpainting, we need to generate a mask
if measure_config['operator']['name'] == 'inpainting':
    mask_gen = mask_generator(**measure_config['mask_opt'])

# Do inference
for i, ref_img in enumerate(loader):

    logger.info(f"Inference for image {i}")
    fname = str(i).zfill(3)
    ref_img = ref_img.to(device)
    set_seed(args.seed)

    # Exception) In case of inpainting
    if measure_config['operator']['name'] == 'inpainting':
        mask = mask_gen(ref_img)
        mask = mask[:, 0, :, :].unsqueeze(dim=0)
        operator_fn = partial(operator.forward, mask=mask)
        sample_fn = partial(sample_fn, operator_fn=operator_fn)

        # Forward measurement model
        y = operator_fn(ref_img)
        y_n = noiser(y)

    else:
        y = operator.forward(ref_img)
        y_n = noiser(y).to(device)

    # Sampling
    x_rec, x_rec_opt = sample_fn(measurement=y_n.detach())

    # Post-processing samples
    label = clear_color(y_n)
    plt.imsave(os.path.join(out_path, task_name, 'label', fname + '_label.png'), label)

    # save original samples
    for j in range(x_rec.shape[0]):
        reconstructed = clear_color(x_rec[j:j + 1, ...])
        plt.imsave(os.path.join(out_path, task_name, 'recon', fname + f'x_{j}_recon.png'), reconstructed)

    # save optimized samples
    if task_name != 'inpainting':
        for j in range(x_rec_opt.shape[0]):
            reconstructed = clear_color(x_rec_opt[j:j + 1, ...])
            plt.imsave(os.path.join(out_path, task_name, 'recon', fname + f'x_{j}_recon_opt.png'), reconstructed)

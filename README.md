# Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo (ICML 2025) - [[Paper]](https://arxiv.org/abs/2502.05908) - Unofficial Implementation
In image processing, solving inverse problems is the task of finding plausible reconstructions of an image that was corrupted by some (usually known) degradation model.
Commonly, this process is done using a generative image model that can guide the reconstruction towards solutions that appear natural. The success of diffusion models over the last few years has made them a leading candidate for this task. However, the sequential nature of diffusion models makes this conditional sampling process challenging. Furthermore, since diffusion models are often defined in the latent space of an autoencoder, the encoder-decoder transformations introduce additional difficulties. Here, we suggest a novel sampling method based on sequential Monte Carlo (SMC) in the latent space of diffusion models. We use the forward process of the diffusion model to add additional auxiliary observations and then perform an SMC sampling as part of the backward process. Empirical evaluations on ImageNet and FFHQ show the benefits of our approach over competing methods on various inverse problem tasks.

### Installation Instructions
1. Install repo:
```bash
conda create -n "LD-SMC" python=3.10
pip install -e .
```

2. Download pretrained checkpoints (autoencoder and model)
```bash
mkdir -p models/ldm/ffhq
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm/ffhq
unzip models/ldm/ffhq.zip -d ./models/ldm/ffhq

mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
```

3. Inference
```bash
python sample_condition
```

### Citation
Please cite this paper if you want to use it in your work,
```
@inproceedings{achituve2025inverse,
title={Inverse Problem Sampling in Latent Space Using Sequential {M}onte {C}arlo},
author={Idan Achituve and Hai Victor Habi and Amir Rosenfeld and Arnon Netzer and Idit Diamant and Ethan Fetaya},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=mWKCajTUUu}
}
```

### External Code Attribution
The code in this repo was adapted from the code base in the following repos: [latent-diffusion](https://github.com/CompVis/latent-diffusion/tree/main), [DPS](https://github.com/DPS2022/diffusion-posterior-sampling), [ReSample](https://github.com/soominkwon/resample), and [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) under the terms of the [license](https://opensource.org/licenses/MIT).

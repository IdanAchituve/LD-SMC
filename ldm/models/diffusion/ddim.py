"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm, trange
from functools import partial
from scripts.utils import *

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from scripts.utils import clear_color
import wandb

from ldm_inverse.svd_replacement import SuperResolution, Deblurring, Deblurring2D


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.out_path = None

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        if ddim_num_steps < 1000:
          ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                    ddim_timesteps=self.ddim_timesteps,
                                                                                    eta=ddim_eta,verbose=verbose)
          self.register_buffer('ddim_sigmas', ddim_sigmas)
          self.register_buffer('ddim_alphas', ddim_alphas)
          self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
          self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
                          (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                           1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
        self.ddim_num_steps = ddim_num_steps

    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        """
        Sampling wrapper function for UNCONDITIONAL sampling.
        """

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        """
        Function for unconditional sampling using DDIM.
        """

        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates


    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec



    def ddecode(self, x_latent, cond=None, t_start=50, temp = 1, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps, temperature = temp, 
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec


class LD_SMC(DDIMSampler):
    def __init__(self, model, schedule="linear", kappa1=1.0, kappa2=3.5,
                 rho=0.75, num_particles=5, s=500, num_gibbs_iters=1, logger=None, **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.rho = rho
        self.num_particles = num_particles
        self.s = s
        self.logger = logger
        self.num_gibbs_iters = num_gibbs_iters

    def pixel_optimization(self, measurement, x_init, operator_fn, eps=1e-5,
                           max_iters=5000):
        """
        Function to compute argmin_x ||y - A(x)||_2^2

        Arguments:
            measurement:           Measurement vector y
            x_init:                initialization of x
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        """

        loss = torch.nn.MSELoss()  # MSE loss

        opt_var = x_init.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=5e-3)  # Initializing optimizer
        measurement = measurement.detach()  # Need to detach for weird PyTorch reasons
        checkpoints = []

        checkpoints.append(torch.clone(opt_var.detach()))
        with torch.no_grad():
            measurement_loss = loss(measurement, operator_fn(opt_var))
            self.logger.info(f'initial pixel loss: {measurement_loss}')

        # Training loop
        for j in range(max_iters):
            optimizer.zero_grad()

            measurement_loss = loss(measurement, operator_fn(opt_var))

            measurement_loss.backward()  # Take GD step
            optimizer.step()

            checkpoints.append(torch.clone(opt_var))

            if j % 100 == 0:
                self.logger.info(f'pixel loss: {measurement_loss} at iter {j}')

            # Convergence criteria
            if measurement_loss < eps ** 2:  # needs tuning according to noise level for early stopping
                break

        return checkpoints

    def latent_optimization(self, measurement, z_init, operator_fn, eps=1e-5,
                            max_iters=500, lr=None):

        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations

        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()

        if lr is None:
            lr_val = 5e-3
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss()  # MSE loss
        optimizer = torch.optim.AdamW([z_init], lr=lr_val)  # Initializing optimizer ###change the learning rate
        measurement = measurement.detach()  # Need to detach for weird PyTorch reasons

        # Training loop
        losses = []
        checkpoints = []

        checkpoints.append(torch.clone(z_init.detach()))
        with torch.no_grad():
            output = loss(measurement, operator_fn(self.model.differentiable_decode_first_stage(z_init)))
            self.logger.info(f'initial latent loss: {output}')

        for itr in range(max_iters):
            optimizer.zero_grad()
            output = loss(measurement, operator_fn(self.model.differentiable_decode_first_stage(z_init)))

            output.backward()  # Take GD step
            optimizer.step()
            cur_loss = output.detach().cpu().numpy()

            if itr % 100 == 0:
                self.logger.info(f'latent loss: {output} at iter {itr}')

            # Convergence criteria
            checkpoints.append(torch.clone(z_init))

            if itr < 200:  # may need tuning for early stopping
                losses.append(cur_loss)
            else:
                losses.append(cur_loss)
                if losses[0] < cur_loss:
                    break
                else:
                    losses.pop(0)

            if cur_loss < eps ** 2:  # needs tuning according to noise level for early stopping
                break

        return checkpoints

    def sample_zs_and_ys_given_y0(self, z_0, y_0, operator_fn,  optimize=True):
        """
        Sample from the distributions auxiliary labels using DDIM
        :param z_0: Approximation to latent variable at time 0 [1, 3, 64, 64]
        :param y_0: The measurement [1, 3, x, x]
        :param operator_fn: the corruption model (without injecting noise)
        """

        ys = []
        zs = []

        b, *_, device = *y_0.shape, y_0.device

        if optimize:
            x_0 = self.model.decode_first_stage(z_0.detach())
            x0_hat = self.pixel_optimization(measurement=y_0,
                                             x_init=x_0.detach(),
                                             operator_fn=operator_fn)[-1]
        else:
            x0_hat = operator_fn(y_0)

        z0_hat = self.model.encode_first_stage(x0_hat.detach())

        alphas = self.model.alphas_cumprod
        alphas_prev = self.model.alphas_cumprod_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod
        sigmas = self.ddim_sigmas_for_original_num_steps

        z_T = torch.randn_like(z_0).to(device)
        zs.append(z_T)
        x_T = self.model.decode_first_stage(z_T).detach()
        ys.append(operator_fn(x_T).detach())

        for i in range(alphas.shape[0] - 1, 0, -1):
            noise = torch.randn_like(z_0).to(device)
            a_t = torch.full((b, 1, 1, 1), alphas[i], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[i], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[i], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[i], device=device)

            z_t = (a_prev.sqrt() * z0_hat + torch.sqrt(1 - a_prev - sigma_t ** 2) *
                   (zs[-1] - a_t.sqrt() * z0_hat) / sqrt_one_minus_at + sigma_t * noise)
            x_t = self.model.decode_first_stage(z_t)
            y_t = operator_fn(x_t)

            ys.append(y_t.detach())
            zs.append(z_t.detach())

        ys.append(y_0.detach())
        return ys[::-1]

    def sample_ys_given_zs(self, zs, y_0, operator_fn):
        """
        Sample from the distributions auxiliary labels using DDIM
        :param zs: [TS, 3, 64, 64]
        :param y_0: The measurement [1, 3, x, x]
        :param operator_fn: the corruption model (without injecting noise)
        """

        ys = []

        b, *_, device = *y_0.shape, y_0.device
        ys.append(y_0.detach())

        for i in range(1, zs.shape[0]):
            x_t = self.model.decode_first_stage(zs[i:i+1, ...])
            y_t = operator_fn(x_t)
            ys.append(y_t.detach())

        return ys

    def p_moments_ddim(self, x, c, t, index, differential=True, quantize_denoised=False,
                       score_corrector=None, corrector_kwargs=None,
                       unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.differential_apply_model(x, t, c) if differential else self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.differential_apply_model(x_in, t_in, c_in).chunk(2) if differential else (
                self.model.apply_model(x_in, t_in, c_in).chunk(2))
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod
        alphas_prev = self.model.alphas_cumprod_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod
        sigmas = self.ddim_sigmas_for_original_num_steps

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        mu_t = a_prev.sqrt() * pred_x0 + dir_xt

        return mu_t.detach(), sigma_t.detach(), pred_x0

    def proposal_dist(self, operator_fn, prior_mean, prior_std, y0, y_t,
                      bar_z0, z_t, step, sample_prior=False):

        num_particles = prior_mean.shape[0]
        if sample_prior:
            return prior_mean.detach(), prior_std.detach()

        f_t_z_0 = operator_fn(self.model.differentiable_decode_first_stage(bar_z0))
        difference = y0.repeat(num_particles, 1, 1, 1) - f_t_z_0
        norm = torch.linalg.norm(difference.view(num_particles, -1), dim=1) ** 2
        g_1 = torch.autograd.grad(outputs=norm.sum(), inputs=z_t)[0]
        g_1 /= torch.maximum(torch.linalg.norm(g_1.view(num_particles, -1), dim=1).view(num_particles, 1, 1, 1),
                             torch.ones((num_particles, 1, 1, 1), device=y0.device))

        if step <= self.s:
            prior_mean_ = torch.clone(prior_mean).requires_grad_()
            difference = (y_t.repeat(num_particles, 1, 1, 1) -
                          operator_fn(self.model.differentiable_decode_first_stage(prior_mean_)))
            norm = torch.linalg.norm(difference.view(num_particles, -1), dim=1) ** 2
            g_2 = torch.autograd.grad(outputs=norm.sum(), inputs=prior_mean_)[0]
            g_2 /= torch.maximum(torch.linalg.norm(g_2.view(num_particles, -1), dim=1).view(num_particles, 1, 1, 1),
                                 torch.ones((num_particles, 1, 1, 1), device=y0.device))
            g = self.kappa2 * ((1 - self.rho) * g_1 + self.rho * g_2)
        else:
            g = self.kappa1 * g_1

        if self.logger is not None:
            self.logger.info(f'mean norm: {torch.linalg.norm(prior_mean.detach()):.4f}')
            self.logger.info(f'update norm: {torch.linalg.norm(g):.6f}')

        return (prior_mean - g).detach(), prior_std.detach()

    def quadratic_term(self, x, m, s):
        avg_dims = (1, 2, 3) if len(x.shape) == 4 else (2, 3, 4) if len(x.shape) == 5 else ()
        return - (1 / 2) * ((1 / s ** 2) * ((x - m) ** 2)).mean(dim=avg_dims)

    @torch.no_grad()
    def get_weights(self, y0, yt, operator_fn, prior_mean, prior_std, prop_samples,
                    bar_z0_t, prev_log_normalized_weights=None, pi=None,
                    index=1000, ll_y0_ztp1=None, first_step=False):
            """
            compute the new importance weights and perform resampling
            :param y_t: The measurement [1, 3, x, x]
            :param operator_fn: the corruption model (without injecting noise)
            :param prior_mean: the mean from the diffusion process p(z_t | z_t+1) [1, 3, x, x]
            :param prior_std: the std from the diffusion process p(z_t | z_t+1)  [1, 3, x, x]
            :param pi: the proposal distribution N(z_t|m, eta * I)
            :param prop_samples: the samples from the proposal distribution  [N * M, 3, x, x]
            :param f_tp_1_z_0: A(D(\bar{z_0}(z_t+1)))
            """
            device = yt.device

            last_step = False if index > 0 else True

            alphas = self.model.alphas_cumprod
            log_prior = self.quadratic_term(prop_samples, prior_mean.detach(), prior_std.detach())

            if first_step:
                a_t = torch.full((1, 1, 1, 1), alphas[index], device=device)
                y_t_std = (1 - a_t).sqrt()
                with torch.no_grad():
                    ft_zt = operator_fn(self.model.decode_first_stage(prop_samples))
                    ll_yt_zt = self.quadratic_term(yt.detach(), ft_zt.detach(), y_t_std)

                    ft_z0 = operator_fn(self.model.decode_first_stage(bar_z0_t))
                    ll_y0_zt = self.quadratic_term(y0.detach(), ft_z0.detach(), y_t_std)
                log_weights = ll_yt_zt + ll_y0_zt

            else:
                a_t = torch.full((1, 1, 1, 1), alphas[index], device=device)
                y_t_std = (1 - a_t).sqrt()

                log_pi = self.quadratic_term(prop_samples, pi.loc.detach(), pi.scale.detach())
                with torch.no_grad():
                    ft_zt = operator_fn(self.model.decode_first_stage(prop_samples))
                    ll_yt_zt = self.quadratic_term(yt.detach(), ft_zt.detach(), y_t_std)
                    if not last_step:
                        ft_z0 = operator_fn(self.model.decode_first_stage(bar_z0_t))
                        ll_y0_zt = self.quadratic_term(y0.detach(), ft_z0.detach(), y_t_std)
                    else:
                        ll_y0_zt = torch.zeros_like(ll_yt_zt, device=ll_yt_zt.device)

                log_weights = (prev_log_normalized_weights + ll_y0_zt +
                               ll_yt_zt + log_prior - ll_y0_ztp1 - log_pi)

            return ll_y0_zt, log_weights

    def ld_smc_sampler(self, measurement, operator_fn, task_name, S, batch_size, shape,
                       conditioning=None, quantize_x0=False, eta=0.,
                       score_corrector=None, corrector_kwargs=None,
                       unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                       verbose=True, **kwargs):
        """
        Sampling wrapper function for inverse problem solving.
        """
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:

                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        self.logger.info(f'Data shape for DDIM sampling is {size}, eta {eta}')

        timesteps = self.ddpm_num_timesteps
        time_range = list(range(0, timesteps))[::-1]
        total_steps = timesteps

        device = measurement.device

        x0 = torch.zeros((self.num_gibbs_iters, 3, 256, 256), device=device)
        x0_opt = torch.zeros((self.num_gibbs_iters, 3, 256, 256), device=device)

        z0 = torch.randn(size, device=device).requires_grad_()
        zs = torch.zeros((total_steps + 1, self.num_particles, C, H, W), device=device)

        for j in range(self.num_gibbs_iters):

            if j == 0:
                ys = self.sample_zs_and_ys_given_y0(z_0=z0, y_0=measurement,
                                                    operator_fn=operator_fn,
                                                    optimize=True)
            else:
                ys = self.sample_ys_given_zs(zs=zs, y_0=measurement, operator_fn=operator_fn)
                zs = torch.zeros((total_steps + 1, self.num_particles, C, H, W), device=device)

            self.logger.info(f"[{j}] sampled y's for all time steps")

            # sample from proposal
            step = time_range[0]
            prior_mean = torch.zeros(size).to(device)
            prior_std = torch.ones(size).to(device)
            proposal_mean, proposal_std = (
                self.proposal_dist(operator_fn, prior_mean.detach(), prior_std, y0=measurement,
                                   y_t=ys[-1], bar_z0=None, z_t=zs[-1], step=step, sample_prior=True)
            )
            pi = torch.distributions.normal.Normal(loc=proposal_mean, scale=proposal_std)
            z_t = pi.sample(sample_shape=torch.Size([self.num_particles]))
            z_t = z_t.permute(1, 0, 2, 3, 4).contiguous().view(-1, *prior_mean.shape[1:])

            # resampling step
            index = total_steps - 1
            ts = torch.full((self.num_particles,), total_steps, device=device, dtype=torch.long)  # assume batch size 1

            z_t.requires_grad_(True)
            mu_t, sigma_t, bar_z0 = self.p_moments_ddim(z_t, conditioning, ts, index=index,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        quantize_denoised=quantize_x0,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs)

            log_normalized_weights = torch.log(torch.ones(self.num_particles, device=device) * (1 / self.num_particles))
            ll_y0_ztp1, log_weights = (
                self.get_weights(y0=measurement, yt=ys[index + 1], operator_fn=operator_fn,
                                 prior_mean=prior_mean.detach(), prior_std=prior_std, prop_samples=z_t,
                                 bar_z0_t=bar_z0, prev_log_normalized_weights=log_normalized_weights, pi=pi,
                                 index=index, ll_y0_ztp1=None, first_step=True))

            zs[total_steps:total_steps+1, ...] = torch.clone(z_t).detach()

            prev_sigma_t = None
            for i, step in enumerate(time_range):
                # Instantiating parameters
                index = total_steps - i - 1
                ts = torch.full((self.num_particles,), step, device=device, dtype=torch.long)  # assume batch size 1

                # make update step for all particles
                proposal_mean, proposal_std = (
                    self.proposal_dist(operator_fn, mu_t, sigma_t, y0=measurement,
                                       y_t=ys[index], bar_z0=bar_z0, z_t=z_t, step=step, sample_prior=False)
                )
                self.logger.info(f"\n[{j}][{index}] - update step")

                # resampling step
                if self.num_particles > 1:
                    prev_time = index + 1
                    # do not resample at the first iteration
                    if prev_time < total_steps:
                        resample_dist = torch.distributions.categorical.Categorical(logits=log_weights)
                        samples_indices = resample_dist.sample(
                            sample_shape=torch.Size([self.num_particles])).squeeze()  # [N, M]

                        # pick particles according to sampling
                        zs_cpy = torch.clone(zs)
                        proposal_mean_cpy = torch.clone(proposal_mean)
                        ll_y0_ztp1_cpy = torch.clone(ll_y0_ztp1)
                        for k, resample_idx in enumerate(samples_indices):
                            zs[:, k, ...] = zs_cpy[:, resample_idx, ...].detach()
                            proposal_mean[k, ...] = proposal_mean_cpy[resample_idx, ...]
                            ll_y0_ztp1[k] = ll_y0_ztp1_cpy[resample_idx]

                        # reset weights
                        log_normalized_weights = torch.log(
                            torch.ones(self.num_particles, device=device) * (1 / self.num_particles))
                    else:
                        log_normalized_weights = log_weights - torch.logsumexp(log_weights, dim=0)

                self.logger.info(f"\n[{j}][{index}] - resampled particles")

                pi = torch.distributions.normal.Normal(loc=proposal_mean, scale=proposal_std)
                z_t = pi.sample(sample_shape=torch.Size([1]))
                z_t = z_t.permute(1, 0, 2, 3, 4).contiguous().view(-1, *prior_mean.shape[1:])

                self.logger.info(f"\n[{j}][{index}] - sampled from proposal")

                # append new sample
                zs[index:index + 1, ...] = torch.clone(z_t).detach()

                z_t.requires_grad_(True)
                mu_t, sigma_t, bar_z0 = self.p_moments_ddim(z_t, conditioning, ts, index=index,
                                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                                            unconditional_conditioning=unconditional_conditioning,
                                                            quantize_denoised=quantize_x0,
                                                            score_corrector=score_corrector,
                                                            corrector_kwargs=corrector_kwargs)

                sigma_t = sigma_t.repeat(1, *mu_t.shape[1:])
                if index == 0:
                    sigma_t = prev_sigma_t  # small noise since in the last step the variance is 0

                self.logger.info(f"\n[{j}][{index}] - calculated diffusion moments")

                # set weights
                if self.num_particles > 1:
                    ll_y0_ztp1, log_weights = (
                        self.get_weights(y0=measurement, yt=ys[index], operator_fn=operator_fn,
                                         prior_mean=torch.clone(mu_t).detach(),
                                         prior_std=torch.clone(sigma_t).detach(),
                                         prop_samples=torch.clone(z_t).detach(),
                                         bar_z0_t=torch.clone(bar_z0).detach(),
                                         prev_log_normalized_weights=log_normalized_weights, pi=pi,
                                         index=index, ll_y0_ztp1=ll_y0_ztp1, first_step=False))

                    if index == 0:
                        max_w, max_w_index = torch.max(log_weights, dim=0)
                        zs = zs[:, max_w_index, ...]
                else:
                    if index == 0:
                        zs = zs[:, 0, ...]

                self.logger.info(f"\n[{j}][{index}] - resampled particles")

                prev_sigma_t = torch.clone(sigma_t).detach()

            z0 = torch.clone(zs[0:1, ...].detach())
            x0[j:j + 1, ...] = self.model.decode_first_stage(torch.clone(z0.detach()))
            z0_opt_ = self.latent_optimization(measurement=measurement.repeat(z0.shape[0], 1, 1, 1),
                                               z_init=torch.clone(z0.detach()),
                                               operator_fn=operator_fn)[-1]
            x0_opt[j:j + 1, ...] = self.model.decode_first_stage(z0_opt_.detach())

        return x0, x0_opt

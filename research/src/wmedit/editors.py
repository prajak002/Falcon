"""Text-based image editors behind one interface: edit(img, step, seed, hook=None) -> PIL.

InstructPix2Pix is run through an explicit denoising loop (a line-for-line port of the diffusers
pipeline) so that guided variants can intervene at each step; `tests/test_editor_equivalence.py`
checks that with no hook it reproduces the official pipeline output.
"""
from __future__ import annotations

from typing import Callable, Protocol

import torch
from PIL import Image

from .data import EditStep
from .utils import get_device, pil_to_tensor, tensor_to_pil

# hook(i, t, latents, x0_latent_pred, sigma) -> new latents
LatentHook = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class Editor(Protocol):
    name: str

    def edit(self, img: Image.Image, step: EditStep, seed: int, hook: LatentHook | None = None,
             **overrides) -> Image.Image: ...


def _fit(img: Image.Image) -> Image.Image:
    w, h = img.size
    w8, h8 = w - w % 8, h - h % 8
    return img if (w8, h8) == (w, h) else img.resize((w8, h8), Image.LANCZOS)


class InstructPix2Pix:
    name = "ip2p"

    def __init__(self, model_id: str = "timbrooks/instruct-pix2pix", steps: int = 50,
                 text_cfg: float = 7.5, image_cfg: float = 1.5, device=None, dtype=torch.float16):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        self.device = device or get_device()
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, variant="fp16" if self.dtype == torch.float16 else None,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.text_cfg, self.image_cfg = steps, text_cfg, image_cfg

    @torch.no_grad()
    def _encode_text(self, prompt: str) -> torch.Tensor:
        # ordered [text, uncond, uncond] as in the diffusers pipeline
        return self.pipe._encode_prompt(prompt, self.device, 1, True, "")

    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None,
             steps: int | None = None, text_cfg: float | None = None, image_cfg: float | None = None):
        p, steps = self.pipe, steps or self.steps
        text_cfg = self.text_cfg if text_cfg is None else text_cfg
        image_cfg = self.image_cfg if image_cfg is None else image_cfg
        img = _fit(img)
        gen = torch.Generator(self.device).manual_seed(seed)
        with torch.no_grad():
            emb = self._encode_text(step.instruction)
            x = (pil_to_tensor(img, self.device) * 2 - 1).to(self.dtype)
            il = p.vae.encode(x).latent_dist.mode()
            il = torch.cat([il, il, torch.zeros_like(il)])
            p.scheduler.set_timesteps(steps, device=self.device)
            ts = p.scheduler.timesteps
            shape = (1, p.vae.config.latent_channels, img.height // 8, img.width // 8)
            lat = torch.randn(shape, generator=gen, device=self.device, dtype=self.dtype)
            lat = lat * p.scheduler.init_noise_sigma
        for i, t in enumerate(ts):
            with torch.no_grad():
                inp = p.scheduler.scale_model_input(torch.cat([lat] * 3), t)
                eps = p.unet(torch.cat([inp, il], 1), t, encoder_hidden_states=emb).sample
                e_txt, e_img, e_unc = eps.chunk(3)
                eps = e_unc + text_cfg * (e_txt - e_img) + image_cfg * (e_img - e_unc)
            if hook is not None:
                sigma = p.scheduler.sigmas[p.scheduler.step_index or i].to(lat)
                lat = hook(i, t, lat, (lat - sigma * eps).detach(), sigma)
            with torch.no_grad():
                lat = p.scheduler.step(eps, t, lat, generator=gen).prev_sample
        return self.decode(lat)

    @torch.no_grad()
    def decode(self, lat: torch.Tensor) -> Image.Image:
        x = self.pipe.vae.decode(lat / self.pipe.vae.config.scaling_factor).sample
        return tensor_to_pil((x.float() + 1) / 2)

    @torch.no_grad()
    def reference_edit(self, img, step: EditStep, seed: int) -> Image.Image:
        """Official diffusers pipeline; used only by the equivalence test."""
        gen = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(step.instruction, image=_fit(img), num_inference_steps=self.steps,
                         guidance_scale=self.text_cfg, image_guidance_scale=self.image_cfg,
                         generator=gen).images[0]


class SDEdit:
    """SDEdit (Meng et al., ICLR 2022) with SD-1.5: partial noising to `strength`, then denoise
    with the target caption."""
    name = "sdedit"

    def __init__(self, model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", steps: int = 50,
                 strength: float = 0.5, cfg: float = 7.5, device=None, dtype=torch.float16):
        from diffusers import StableDiffusionImg2ImgPipeline

        self.device = device or get_device()
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, variant="fp16" if self.dtype == torch.float16 else None,
            safety_checker=None, requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.strength, self.cfg = steps, strength, cfg

    @torch.no_grad()
    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None,
             strength: float | None = None, **_):
        if hook is not None:
            raise NotImplementedError("guided SDEdit not implemented")
        gen = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(step.target_caption, image=_fit(img), strength=strength or self.strength,
                         num_inference_steps=self.steps, guidance_scale=self.cfg,
                         generator=gen).images[0]


class Flux2Klein:
    """FLUX.2 [klein] 4B (Black Forest Labs, 2026; Apache-2.0): rectified-flow transformer with reference-image
    editing. Distilled: 4 steps, guidance 1.0 (model card). Takes the same instruction as InstructPix2Pix and runs
    at the input resolution (512x512) so that resizing never acts as an extra attack."""
    name = "flux2klein"

    def __init__(self, model_id: str = "black-forest-labs/FLUX.2-klein-4B", steps: int = 4, guidance: float = 1.0,
                 device=None, cpu_offload: bool = False):
        from diffusers import Flux2KleinPipeline

        self.device = device or get_device()
        dt = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dt)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.guidance = steps, guidance

    @torch.no_grad()
    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None, **_):
        if hook is not None:
            raise NotImplementedError("guided FLUX.2 editing not implemented")
        img = _fit(img)
        gen = torch.Generator(self.device).manual_seed(seed)
        return self.pipe(image=img, prompt=step.instruction, height=img.height, width=img.width,
                         num_inference_steps=self.steps, guidance_scale=self.guidance, generator=gen).images[0]


class TurboEdit:
    """TurboEdit (Deutch et al., SIGGRAPH Asia 2024, arXiv:2408.00735) on SDXL-Turbo, implemented from the paper
    (the official repo carries no licence). Edit-friendly DDPM inversion made few-step by (i) a timestep shift
    Delta for the network/scheduler (Eq. 6-7), (ii) norm-clipping the last correction, (iii) pseudo-guidance w on
    the cross-prompt term (Eq. 10):
        x^_{s} = x_s + mu(x^_t, c) - mu(x_t, c) + w (mu(x^_t, c^) - mu(x^_t, c)),   all mu evaluated at t + Delta.
    Here x_s = sqrt(a_s) x0 + sqrt(1 - a_s) eps_s is the *closed-form* noising of the input (Eq. 3), so the
    watermarked input is carried through every step and only the correction is generated. That makes it the
    editor where the regeneration thesis predicts the *highest* watermark survival at a given edit magnitude.
    Captions, not instructions: c = source caption, c^ = target caption. Defaults from Sec. 4.4:
    4 steps from t = 599, Delta = 200, w = 1.5, clip 15.5."""
    name = "turboedit"

    def __init__(self, model_id: str = "stabilityai/sdxl-turbo", steps: int = 4, t_start: int = 599, delta: int = 200,
                 w: float = 1.5, clip_norm: float = 15.5, device=None, dtype=torch.float16):
        from diffusers import AutoPipelineForText2Image

        self.device = device or get_device()
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=self.dtype, variant="fp16" if self.dtype == torch.float16 else None).to(self.device)
        self.pipe.vae.to(torch.float32)  # the SDXL VAE overflows in fp16
        self.pipe.set_progress_bar_config(disable=True)
        self.ac = self.pipe.scheduler.alphas_cumprod.to(self.device, torch.float32)
        self.steps, self.t_start, self.delta, self.w, self.clip_norm = steps, t_start, delta, w, clip_norm

    def _timesteps(self, steps: int) -> list[int]:
        # t_start -> ... -> 0, equally spaced; transition i goes from ts[i] to ts[i+1]
        return [round(self.t_start * (1 - i / steps)) for i in range(steps)] + [0]

    @torch.no_grad()
    def _cond(self, prompt: str, h: int, w: int):
        p = self.pipe
        emb, _, pooled, _ = p.encode_prompt(prompt, device=self.device, num_images_per_prompt=1,
                                            do_classifier_free_guidance=False)
        tid = p._get_add_time_ids((h, w), (0, 0), (h, w), dtype=emb.dtype,
                                  text_encoder_projection_dim=p.text_encoder_2.config.projection_dim).to(self.device)
        return emb, {"text_embeds": pooled, "time_ids": tid}

    def _x0(self, x: torch.Tensor, t: int, emb, added) -> torch.Tensor:
        """Network at the shifted timestep t + Delta; x0 recovered with the same shifted alpha (Eq. 6)."""
        tt = min(t + self.delta, len(self.ac) - 1)
        eps = self.pipe.unet(x.to(self.dtype), torch.tensor([tt] * len(x), device=self.device),
                             encoder_hidden_states=emb, added_cond_kwargs=added).sample.float()
        a = self.ac[tt]
        return (x - (1 - a).sqrt() * eps) / a.sqrt()

    def _mu(self, x0: torch.Tensor, t_from: int, s: int) -> torch.Tensor:
        """Posterior mean of the skipped DDPM transition t_from(+Delta) -> s given the x0 estimate."""
        if s == 0:
            return x0
        a_t, a_s = self.ac[min(t_from + self.delta, len(self.ac) - 1)], self.ac[s]
        var = (1 - a_s) / (1 - a_t) * (1 - a_t / a_s)
        eps = (self._xt - a_t.sqrt() * x0) / (1 - a_t).sqrt()  # consistent with how x0 was recovered
        return a_s.sqrt() * x0 + (1 - a_s - var).clamp_min(0).sqrt() * eps

    @torch.no_grad()
    def edit(self, img, step: EditStep, seed: int, hook: LatentHook | None = None,
             steps: int | None = None, w: float | None = None, **_):
        if hook is not None:
            raise NotImplementedError("guided TurboEdit not implemented")
        p, steps, w = self.pipe, steps or self.steps, self.w if w is None else w
        img = _fit(img)
        gen = torch.Generator(self.device).manual_seed(seed)
        x = pil_to_tensor(img, self.device).float() * 2 - 1
        x0 = p.vae.encode(x).latent_dist.mode() * p.vae.config.scaling_factor
        ts = self._timesteps(steps)
        noise = lambda: torch.randn(x0.shape, generator=gen, device=self.device)  # noqa: E731
        xs = {t: self.ac[t].sqrt() * x0 + (1 - self.ac[t]).sqrt() * noise() for t in ts[:-1]}  # Eq. 3
        xs[0] = x0
        emb_c, add_c = self._cond(step.source_caption, img.height, img.width)
        emb_e, add_e = self._cond(step.target_caption, img.height, img.width)
        xhat = xs[ts[0]]
        for i, (t, s) in enumerate(zip(ts[:-1], ts[1:])):
            batch = torch.cat([xs[t], xhat, xhat])
            emb = torch.cat([emb_c, emb_c, emb_e])
            added = {k: torch.cat([add_c[k], add_c[k], add_e[k]]) for k in add_c}
            x0_src, x0_hat_c, x0_hat_e = self._x0(batch, t, emb, added).chunk(3)
            self._xt = xs[t]
            mu_src = self._mu(x0_src, t, s)
            self._xt = xhat
            mu_hat_c, mu_hat_e = self._mu(x0_hat_c, t, s), self._mu(x0_hat_e, t, s)
            corr = xs[s] - mu_src  # the inverted "noise" sigma_t z_t (Eq. 4), never sampled
            if i == len(ts) - 2 and self.clip_norm:
                n = corr.flatten(1).norm(dim=1).clamp_min(1e-8)
                corr = corr * (n.clamp_max(self.clip_norm) / n).view(-1, 1, 1, 1)
            xhat = mu_hat_c + corr + w * (mu_hat_e - mu_hat_c)  # Eq. 10
        out = p.vae.decode(xhat / p.vae.config.scaling_factor).sample
        return tensor_to_pil((out.float().clamp(-1, 1) + 1) / 2)


REGISTRY = {"ip2p": InstructPix2Pix, "sdedit": SDEdit, "flux2klein": Flux2Klein, "turboedit": TurboEdit}


def build_editor(name: str, **kwargs) -> Editor:
    return REGISTRY[name](**kwargs)

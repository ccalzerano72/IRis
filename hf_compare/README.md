---
title: IRis-compare
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: gradio
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Degrade, restore and compare blind image restoration engines
tags:
  - image
  - restoration
  - comparison
  - metrics
  - diffusion
models:
  - ccalzerano72/IRis-hybrid-003
  - deepinv/Restormer
  - sd2-community/stable-diffusion-2-1
preload_from_hub:
  - ccalzerano72/IRis-hybrid-003
  - sd2-community/stable-diffusion-2-1 vae/config.json,vae/diffusion_pytorch_model.safetensors,text_encoder/config.json,text_encoder/model.safetensors,text_encoder/model.fp16.safetensors,scheduler/scheduler_config.json,tokenizer/merges.txt,tokenizer/special_tokens_map.json,tokenizer/tokenizer_config.json,tokenizer/vocab.json
  - deepinv/Restormer real_denoising.pth
---

# IRis-compare — Restore & Compare

Companion demo of [IRis](https://huggingface.co/spaces/ccalzerano72/IRis):
upload a clean image, degrade it (presets or thesis-style two-stage pipeline),
restore it with **IRis**, **Restormer** (Real_Denoising) and **Real-ESRGAN**
(x4plus, outscale 1), and compare full-reference (PSNR/SSIM/LPIPS) and
no-reference (NIQE/MUSIQ) metrics — the same protocol as Chapter 5 of the
[IRis thesis](https://github.com/ccalzerano72/IRis).

## About IRis

**IRis** (Image Restoration via latent diffusion) is a Master's thesis project
(University of Pisa, by Carmelo Calzerano) that repurposes Stable Diffusion 2
for blind image restoration via a dual-conditioned latent diffusion
architecture: an 8-channel UNet conditioned on the degraded image in latent
space plus a ControlNet branch providing pixel-space structural guidance,
trained jointly on synthetic degradations. It ranks 1st on all metrics
(PSNR/SSIM/ΔE/LPIPS) on both synthetic (DIV2K) and real-world (RealSR)
degradations against Real-ESRGAN, Restormer, DiffBIR and HyPIR.

- 🖼️ Single-image demo: [ccalzerano72/IRis](https://huggingface.co/spaces/ccalzerano72/IRis)
- 💻 Project repository: [github.com/ccalzerano72/IRis](https://github.com/ccalzerano72/IRis)
- ⚖️ Model weights: [ccalzerano72/IRis-hybrid-003](https://huggingface.co/ccalzerano72/IRis-hybrid-003)

## How this Space works

All engines run locally in this Space (IRis and Restormer on GPU, Real-ESRGAN
on CPU). The Real-ESRGAN weights (`weights/RealESRGAN_x4plus.pth`) are bundled
in this repository; IRis and Restormer weights are preloaded from the Hub.

- IRis weights: [ccalzerano72/IRis-hybrid-003](https://huggingface.co/ccalzerano72/IRis-hybrid-003)
- Restormer weights: [deepinv/Restormer](https://huggingface.co/deepinv/Restormer)
- Project repository: [github.com/ccalzerano72/IRis](https://github.com/ccalzerano72/IRis)

## License

IRis model weights: OpenRAIL++-M (see `LICENSE` in the
[model repository](https://huggingface.co/ccalzerano72/IRis-hybrid-003)),
including the use restrictions in Attachment A. App code: Apache-2.0.

## Test data

One-click examples in the app come from standard super-resolution benchmarks
(resized to ≤768px):

- **Urban100** (Huang et al., CVPR 2015) — official source:
  [jbhuang0604/SelfExSR](https://github.com/jbhuang0604/SelfExSR);
  HF mirror: [eugenesiow/Urban100](https://huggingface.co/datasets/eugenesiow/Urban100)
  (images under CC-BY-4.0, attribution: Huang, Singh & Ahuja, 2015).
- **DIV2K valid** (Agustsson et al., NTIRE 2017) — official source:
  [data.vision.ee.ethz.ch/cvl/DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/);
  HF mirror: [eugenesiow/Div2k](https://huggingface.co/datasets/eugenesiow/Div2k)
  (research use; please cite the NTIRE 2017 challenge paper).
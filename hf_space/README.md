---
title: IRis
emoji: 🖼️
colorFrom: indigo
colorTo: purple
sdk: gradio
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Blind image restoration via latent diffusion
tags:
  - image
  - restoration
  - diffusion
  - controlnet
  - stable-diffusion
models:
  - ccalzerano72/IRis-hybrid-003
  - sd2-community/stable-diffusion-2-1
preload_from_hub:
  - ccalzerano72/IRis-hybrid-003
  - sd2-community/stable-diffusion-2-1 vae/config.json,vae/diffusion_pytorch_model.safetensors,text_encoder/config.json,text_encoder/model.safetensors,text_encoder/model.fp16.safetensors,scheduler/scheduler_config.json,tokenizer/merges.txt,tokenizer/special_tokens_map.json,tokenizer/tokenizer_config.json,tokenizer/vocab.json
---

# IRis — Blind Image Restoration

Master's thesis demo (University of Pisa): **"Blind Image Restoration via Dual-Conditioned Latent Diffusion"** by Carmelo Calzerano.

IRis extends the [Marigold](https://github.com/prs-eth/Marigold) framework to image restoration via:

1. an **8-channel UNet** conditioning the denoising process on the degraded image in latent space
2. a **ControlNet** branch providing pixel-space structural guidance
3. **joint training** on synthetic degradations (noise, blur, JPEG, resize) from a LAION-Aesthetics subset

The app runs on **ZeroGPU** shared hardware. The model is streamed to the GPU on first use, so the initial request may take longer.

- Model weights repo: [ccalzerano72/IRis-hybrid-003](https://huggingface.co/ccalzerano72/IRis-hybrid-003)
- Project repository: [github.com/ccalzerano72/IRis](https://github.com/ccalzerano72/IRis)
- Comparison demo: [ccalzerano72/IRis-compare](https://huggingface.co/spaces/ccalzerano72/IRis-compare)

## License

Model weights: OpenRAIL++-M (see `LICENSE` in the
[model repository](https://huggingface.co/ccalzerano72/IRis-hybrid-003)),
including the use restrictions in Attachment A. App code: Apache-2.0.

## Test data

One-click examples in the app (pre-degraded with the Medium preset) come from
standard super-resolution benchmarks:

- **Urban100** (Huang et al., CVPR 2015) — official source:
  [jbhuang0604/SelfExSR](https://github.com/jbhuang0604/SelfExSR);
  HF mirror: [eugenesiow/Urban100](https://huggingface.co/datasets/eugenesiow/Urban100)
  (images under CC-BY-4.0, attribution: Huang, Singh & Ahuja, 2015).
- **DIV2K valid** (Agustsson et al., NTIRE 2017) — official source:
  [data.vision.ee.ethz.ch/cvl/DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/);
  HF mirror: [eugenesiow/Div2k](https://huggingface.co/datasets/eugenesiow/Div2k)
  (research use; please cite the NTIRE 2017 challenge paper).
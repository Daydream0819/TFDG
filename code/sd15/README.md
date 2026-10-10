# Stable Diffusion v1.5 epsilon-prediction adapter

This adapter applies all clean-anchor guidance rules to the SD1.5 U-Net's
native noise prediction. Each solver step follows

```text
epsilon prediction -> clean latent anchor -> guidance -> epsilon prediction
-> DPM-Solver++ update.
```

The default is 25 DPM-Solver++ steps at 512x512. This is the formal setting
for the SD1.5 migration experiment; lower step counts may be used only for
development-set candidate screening and must be reconfirmed at 25 steps.

The package uses the public `runwayml/stable-diffusion-v1-5` Diffusers model.
Set `HF_ENDPOINT=https://hf-mirror.com` before setup when the server cannot
reach Hugging Face directly.

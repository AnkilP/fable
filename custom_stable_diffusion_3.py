from diffusers import DiffusionPipeline


class CustomStableDiffusion3PipeLine(DiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, prompt, *args, **kwargs):
        # Run Stable Diffusion normally
        output = super().__call__(prompt, **kwargs)

        # Extract latents before decoding
        latents = self.vae.encode(output).latent_dist.sample()

        return latents, output.images

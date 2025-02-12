# ---
# output-directory: "/tmp/stable-diffusion"
# args: ["--prompt", "A 1600s oil painting of the New York City skyline"]
# ---

# # Run Stable Diffusion 3.5 Large Turbo as a CLI, API, and web UI

# This example shows how to run [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) on Modal
# to generate images from your local command line, via an API, and as a web UI.

# Inference takes about one minute to cold start,
# at which point images are generated at a rate of one image every 1-2 seconds
# for batch sizes between one and 16.

# Below are four images produced by the prompt
# "A princess riding on a pony".

# ![stable diffusion montage](https://modal-cdn.com/cdnbot/sd-montage-princess-yxu2vnbl_e896a9c0.webp)

# ## Basic setup

import io
import random
import time
from pathlib import Path

import modal

MINUTES = 60

# All Modal programs need an [`App`](https://modal.com/docs/reference/modal.App) — an object that acts as a recipe for
# the application. Let's give it a friendly name.

app = modal.App("fable")

# ## Configuring dependencies

# The model runs remotely inside a [container](https://modal.com/docs/guide/custom-container).
# That means we need to install the necessary dependencies in that container's image.

# Below, we start from a lightweight base Linux image
# and then install our Python dependencies, like Hugging Face's `diffusers` library and `torch`.

CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
        "DeepCache==0.1.1"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE_DIR": CACHE_DIR,
        }
    )
)

with image.imports():
    import diffusers
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    import torch
    from fastapi import Response
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from huggingface_hub._login import login
    from DeepCache import DeepCacheSDHelper
    import numpy as np
    from io import BytesIO
    import torchvision.transforms as T
    from tqdm.auto import tqdm
    from PIL import Image

# ## Implementing SD3.5 Large Turbo inference on Modal

# We wrap inference in a Modal [Cls](https://modal.com/docs/guide/lifecycle-methods)
# that ensures models are loaded and then moved to the GPU once when a new container
# starts, before the container picks up any work.

# The `run` function just wraps a `diffusers` pipeline.
# It sends the output image back to the client as bytes.

# We also include a `web` wrapper that makes it possible
# to trigger inference via an API call.
# See the `/docs` route of the URL ending in `inference-web.modal.run`
# that appears when you deploy the app for details.

# MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    volumes={CACHE_DIR: cache_volume},
    keep_warm=1,
    container_idle_timeout=12 * MINUTES,
)
class Inference:
    @modal.enter()
    def load_pipeline(self):
        model_name = "microsoft/phi-4"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto"  # Automatically selects GPU if available
        )

        # Create text generation pipeline
        self.llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            # revision=MODEL_REVISION_ID,
            torch_dtype=torch.float16,
            vae=self.vae,
        ).to("cuda")
        self.pipe.enable_vae_tiling()
        self.pipe.enable_vae_slicing()
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)
        self.helper = DeepCacheSDHelper(pipe=self.pipe)
        self.helper.set_params(
            cache_interval=3,
            cache_branch_id=0,
        )
        self.prompts = []
        self.gif = False
        self.transform = T.ToPILImage()

    @modal.method()
    def run(
            self, prompt: str, seed: int = None
    ):
        seed = seed if seed is not None else random.randint(0, 2 ** 32 - 1)
        torch.manual_seed(seed)
        prompt_template = f"{prompt}. Keep the resulting text to 70 tokens. Enhance this text-to-image prompt for Stable Diffusion 3. Paul Atreides has just realized he is the Kwisatz Haderach. Infinite possibilities abound. Help create scenes and plots around the user's prompts. Add high detail, make it clear and crisp and make it more cinematic. Enhanced prompt:"
        enhanced = self.llm(prompt_template, max_length=150, do_sample=True)
        new_prompt = enhanced[0]['generated_text']

        latents = torch.randn(
            (2, self.pipe.unet.config.in_channels, 512 // 8, 512 // 8),
            generator=torch.manual_seed(seed), dtype=torch.float16,
        )
        # slerp

        def slerp(v0, v1, num, t0=0, t1=1):
            v0 = v0.detach().cpu().numpy()
            v1 = v1.detach().cpu().numpy()

            def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
                """helper function to spherically interpolate two arrays v1 v2"""
                dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
                if np.abs(dot) > DOT_THRESHOLD:
                    v2 = (1 - t) * v0 + t * v1
                else:
                    theta_0 = np.arccos(dot)
                    sin_theta_0 = np.sin(theta_0)
                    theta_t = theta_0 * t
                    sin_theta_t = np.sin(theta_t)
                    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
                    s1 = sin_theta_t / sin_theta_0
                    v2 = s0 * v0 + s1 * v1
                return v2

            t = np.linspace(t0, t1, num)

            v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]), dtype=torch.float16)

            return v3

        interpolated_latents = slerp(latents[0], latents[1], 10)
        images = []
        self.helper.enable()
        for latent_vector in tqdm(interpolated_latents):
            images.append(
                self.pipe(
                    new_prompt,
                    height=512,
                    width=512,
                    negative_prompt="poorly drawn,cartoon, 3d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry",
                    num_images_per_prompt=1,
                    num_inference_steps=15,
                    guidance_scale=8,
                    generator=torch.manual_seed(seed),
                    latents=latent_vector[None, ...],
                    dtype=torch.float16,
                ).images
            )
        self.helper.disable()
        interpolated_images = [
            Image.fromarray(np.array(image[0], dtype=np.uint8)) for image in images
        ]

        gif_buffer = BytesIO()
        interpolated_images[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=interpolated_images[1:],
            duration=100,  # 100ms per frame
            loop=0
        )
        print("gif")
        gif_buffer.seek(0)
        return gif_buffer.getvalue()

    @modal.web_endpoint(docs=True)
    def web(self, prompt: str, seed: int = None):
        return Response(
            content=self.run.local(  # run in the same container
                prompt, seed=seed
            ),
            media_type="image/gif",
        )


# ## Generating Stable Diffusion images from the command line

# This is the command we'll use to generate images. It takes a text `prompt`,
# a `batch_size` that determines the number of images to generate per prompt,
# and the number of times to run image generation (`samples`).

# You can also provide a `seed` to make sampling more deterministic.

# Run it with

# ```bash
# modal run text_to_image.py
# ```

# and pass `--help` to see more options.


@app.local_entrypoint()
def entrypoint(
        samples: int = 4,
        prompt: str = "A princess riding on a pony",
):
    print(
        f"prompt => {prompt}",
        f"samples => {samples}",
        sep="\n",
    )

    output_dir = Path("/tmp/stable-diffusion")
    output_dir.mkdir(exist_ok=True, parents=True)

    inference_service = Inference()

    for sample_idx in range(samples):
        start = time.time()
        images = inference_service.run.remote(prompt, None)
        duration = time.time() - start
        print(f"Run {sample_idx + 1} took {duration:.3f}s")
        if sample_idx:
            print(
                f"\tGenerated {len(images)} image(s) at {(duration) / len(images):.3f}s / image."
            )
        for batch_idx, image_bytes in enumerate(images):
            output_path = (
                    output_dir
                    / f"output_{slugify(prompt)[:64]}_{str(sample_idx).zfill(2)}_{str(batch_idx).zfill(2)}.png"
            )
            if not batch_idx:
                print("Saving outputs", end="\n\t")
            print(
                output_path,
                end="\n" + ("\t" if batch_idx < len(images) - 1 else ""),
            )
            output_path.write_bytes(image_bytes)


# ## Generating Stable Diffusion images via an API

# The Modal `Cls` above also included a [`web_endpoint`](https://modal.com/docs/examples/basic_web),
# which adds a simple web API to the inference method.

# To try it out, run

# ```bash
# modal deploy text_to_image.py
# ```

# copy the printed URL ending in `inference-web.modal.run`,
# and add `/docs` to the end. This will bring up the interactive
# Swagger/OpenAPI docs for the endpoint.

# ## Generating Stable Diffusion images in a web UI

# Lastly, we add a simple front-end web UI (written in Alpine.js) for
# our image generation backend.

# This is also deployed by running

# ```bash
# modal deploy text_to_image.py.
# ```

# The `Inference` class will serve multiple users from its own auto-scaling pool of warm GPU containers automatically.

frontend_path = Path(__file__).parent / "frontend"

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jinja2==3.1.4", "fastapi[standard]==0.115.4")
    .add_local_dir(frontend_path, remote_path="/assets")
)


@app.function(
    image=web_image,
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def ui():
    import fastapi.staticfiles
    from fastapi import FastAPI, Request
    from fastapi.templating import Jinja2Templates

    web_app = FastAPI()
    templates = Jinja2Templates(directory="/assets")

    @web_app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "inference_url": Inference.web.web_url,
                "model_name": "Choose your own adventure",
                "default_prompt": "You are Paul Atreides. You have just realized you are the Kwisatz Haderach. Infinite possibilities abound. What will you do?",
            },
        )

    web_app.mount(
        "/static",
        fastapi.staticfiles.StaticFiles(directory="/assets"),
        name="static",
    )

    return web_app


def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")

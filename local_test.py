from optimized_stable_diffusion import OptimizedStableDiffusion3Pipeline
import diffusers
import torch

MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION_ID,
    torch_dtype=torch.bfloat16,
)
pipe = OptimizedStableDiffusion3Pipeline(pipe)
output = pipe(prompt="hello")

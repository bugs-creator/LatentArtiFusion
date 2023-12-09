import os
import argparse


import torch
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from torchvision import transforms

from model_training.scripts.improved_diffusion.script_util import create_model
from pipeline_latent_diffusion import LatentDiffusionPipeline


parser = argparse.ArgumentParser()

parser.add_argument('--vae', type=str, help='VAE path')
parser.add_argument('--unet', type=str,help='Unet path')
parser.add_argument('--seed',type=int,default=-1,help="generate seed")

args = parser.parse_args()


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# ol_image=Image.open("D:\Projects\ArtiFusion-20231114T061723Z-001\ArtiFusion\Training_data\\trainB\G1900390 - 2020-10-14 31.06.52_23040_34816.png")
#
# augmentations = transforms.Compose(
#     [
#         transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.CenterCrop(256) if True else transforms.RandomCrop(256),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )
#
#
# def transform_images(examples):
#     images = [augmentations(image.convert("RGB")) for image in examples]
#     return {"input": images}
#
# images=transform_images([ol_image])
#
# images=images["input"][0].unsqueeze(0)
# images=images.to("cuda:0")








vae = AutoencoderKL.from_pretrained(
    args.vae, revision=False
)



model=UNet2DModel.from_pretrained(args.unet,subfolder="unet")

vae.requires_grad_(False)
model.requires_grad_(False)

vae.to("cuda:0")
model.to("cuda:0")

noise_scheduler=DDPMScheduler.from_pretrained("./scheduler")

pipeline = LatentDiffusionPipeline(
        vae=vae,
        unet=model,
        scheduler=noise_scheduler
    )
generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
# run pipeline in inference (sample random noise and denoise)
image = pipeline(num_inference_steps=50, generator=generator).images[0]

image.show()

pass
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
import os
import copy
import glob
from torchvision import transforms
import argparse

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# function to create dir
def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

# function to convert numpy array to pil
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

# main part
def main():
    # read input image path and vae path
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help="path to the image") 
    parser.add_argument('--vae_path', required=True, help="path to the vae") 

    args = parser.parse_args()

    image_path = args.image_path
    vae_path = args.vae_path
    # get all png images under given path
    image_list = glob.glob(os.path.join(image_path,"*.png"))
    # print(image_list)

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples]
        return {"input": images}

    # load pretrained vae
    vae = AutoencoderKL.from_pretrained(
        vae_path, revision=False,map_location=device
    )

    vae.requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(0)

    augmentations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256) if True else transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # loop all images, and reconstruct they
    for imgP in image_list:

        image_name = image_path.split('/')[-1]
        ol_image=Image.open(imgP)

        images=transform_images([ol_image])
   

        images=images["input"][0].unsqueeze(0)

        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor


        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
        image = numpy_to_pil(image)[0]

        # save the decoded results from pretrained var
        save_name = imgP.replace(".png","_afterVAE" + '.png')
        image.save(save_name)
    # image.show()

if __name__ == "__main__":
    main()


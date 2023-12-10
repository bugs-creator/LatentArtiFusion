import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
import os
import copy
from torchvision import transforms
import argparse

# check device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# numpy array to pil
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

def main():
    # get paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',required=True, help="path to the image") 
    parser.add_argument("--save_path",required=True)
    parser.add_argument("--vae_path",required=True)
    args = parser.parse_args()
    
    image_path = args.image_path
    # get image name
    image_name = image_path.split('/')[-1]
    # get mask path by replacing histology_artifacts by histology_mask
    mask_path = image_path.replace("histology_artifacts","histology_mask")
    save_path = args.save_path
    vae_path = args.vae_path

    vae = AutoencoderKL.from_pretrained(
        vae_path, revision=False,map_location=device
    )

    vae.requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(0)
    
    # open the image and mask
    ol_image=Image.open(image_path)
    mask = Image.open(mask_path)

    augmentations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256) if True else transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples]
        return {"input": images}

    # transfer
    images=transform_images([ol_image])
    masks=transform_images([mask])

    images=images["input"][0].unsqueeze(0)
    masks=masks["input"][0].unsqueeze(0)

    # feed them into the latent space
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    latents_mask = vae.encode(masks).latent_dist.sample()
    latents_mask = latents_mask * vae.config.scaling_factor

    # tensor=latents[0][3]
    # latent space sized at 4x32x32, we visualized different dimension 
    for i in range(4):
        tensor=latents[0][i]
        tensor_mask=latents_mask[0][i]

        # normalization
        tensor_normalized = ((tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255).int()
        tensor_mask_normalized = ((tensor_mask - tensor_mask.min()) / (tensor_mask.max() - tensor_mask.min()) * 255).int()

        
        _image=np.stack([tensor_normalized.numpy(),tensor_normalized.numpy(),tensor_normalized.numpy()]).transpose(1,2,0).astype(np.uint8)
        _mask=np.stack([tensor_mask_normalized.numpy(),tensor_mask_normalized.numpy(),tensor_mask_normalized.numpy()]).transpose(1,2,0).astype(np.uint8)
        _mask[_mask<125] = 0
        _mask[_mask!=0] = 255
        _masked_img = _image.copy()
        _masked_img[_mask == 0] = 0

        image = Image.fromarray(_image)
        mask = Image.fromarray(_mask)
        masked_img  = Image.fromarray(_masked_img)

        # save images
        save_name_mask = os.path.join(save_path,image_name.replace(".png","_mask"+str(i) + '.png'))
        save_name_mask_img = os.path.join(save_path,image_name.replace(".png","_maskedImg_"+str(i) + '.png'))
        save_name = os.path.join(save_path,image_name.replace(".png","_"+str(i) + '.png'))
        image.save(save_name)
        masked_img.save(save_name_mask_img)
        mask.save(save_name_mask)

    # decode back and save decoded

    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
    image = numpy_to_pil(image)[0]

    save_name = os.path.join(save_path,image_name.replace(".png","_afterVAE" + '.png'))
    image.save(save_name)
    # image.show()

if __name__ == "__main__":
    main()


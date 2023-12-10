# Latent ArtiFusion

This is a latent space medical image restoration implementation based on https://github.com/zhenqi-he/ArtiFusion Engineering. The project is trained and tested on the diffusers framework. Based on **Artifact Restoration in Histology Images with Diffusion Probabilistic Models** [paper](https://arxiv.org/abs/2307.14262), we propose a novel latent diffusion reconstruction framework and implement a unique generated pipeline in `pipeline_latent_diffusion.py`.

## Quick Start

- [Set-up](#Setup)
- [Load Our Trained Weights](#LoadPretrained)
- [Train your own Model](#Self-Train)
- [Evaluation](#Evaluation)
- [Validation of Pre-trained VAE](#Validation_Pretrained-VAE)
- [Acknowledgement](#Acknowledgement)


## Setup

First, make sure you have **torch 2+ with cuda** (Important! We only achieved GPU version) included in your environment, then install the environment:

```
pip install -r requirements.txt
```

Then generate the default config of training:

```
accelerate config default
```

Then download the pre-trained VAE from [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/vae). Put the file in a directory as below:
(For manually downloading all files in the folder named vae, you may need to rename the downloaded var_config.json to config.json)
```
vae
├─config.json
├─diffusion_pytorch_model.bin
├─diffusion_pytorch_model.fp16.bin
├─diffusion_pytorch_model.fp16.safetensors
└─diffusion_pytorch_model.safetensors
```

## LoadPretrained


You can download our pretrained unet model from the google drive [link](https://drive.google.com/file/d/1actPH17G3ksi051_hsGTIVSJeqTL0BbH/view?usp=sharing), then unzip it and put it in a folder.

The following script will generate an image randomly.

```
python generate_image.py 
--vae=<vae dir> 
--unet=<unet dir>
--seed=<random seed>
```

You can repair an image using the following script. There are some example files in `\examples` folder.

```
python generate_impainting.py
--vae=<vae dir> 
--unet=<unet dir>
--seed=<random seed>
--image=<image path> 
--mask=<mask path>
```

## Self-Train

You can train a new model from scratch. We provide two training codes for unet. Before the training begins, you need to complete the steps of the Environment Setup.

You can download our dataset from the Google Drive [link](https://drive.google.com/drive/folders/13SDZzcgtO3RIdZIiteb-jo3hUtGAqOuq). Our model is trained on `Training_data/trainB`. You can also generate your own dataset by putting images in a directory.

Here is the launch script for training.

```
accelerate launch train_latent_diffusion.py \
--train_data_dir=<dataset dir> \
--output_dir=<output dir> \
--vae_dir=<vae dir> \
--resolution=256 \
--center_crop \
--random_flip \
--train_batch_size=16 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--mixed_precision="fp16" \
--learning_rate=1e-04 \
--max_grad_norm=1 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_train_epochs=1600 \
--checkpointing_steps=2000 \
--dataloader_num_workers=1
```

If you decide to continue your last train, you can add the following args:

```
--resume_from_checkpoint=<checkpoint dir> 
```

We also provide a training script on another kind of UNet. But it's less effective on latent space. Try it if you're interested.

```
accelerate launch train_latent_swinunet.py \
--train_data_dir=<dataset dir> \
--output_dir=<output dir> \
--vae_dir=<vae dir> \
--resolution=256 \
--center_crop \
--random_flip \
--train_batch_size=16 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--mixed_precision="fp16" \
--learning_rate=1e-04 \
--max_grad_norm=1 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_train_epochs=1600 \
--checkpointing_steps=2000 \
--dataloader_num_workers=1
```

During the train, you can use the tensorboard to monitor the train process:

```
tensorboard --logdir="<output_dir>/logs"
```
## Evaluation

To evaluate the reconstruction quality, we test our model on an artifact-free histology dataset with a total of 462 images sized at 256X256 and then compare the image similarities between reconstructed images and original images to evaluate the performance. 

To evaluate your model performance, modify the relevant paths in test/eval.sh and then run:

```
cd test
sh eval.sh
```

## Validation_Pretrained-VAE
To reproduce our validation experiments of pre-trained VAE for medical histology images, firstly select some test images and put they all into a folder, then replace the PATH_TO_IAMGES in the following commands by the path to the test folders and also replace the PATH_TO_VAE by the downloaded pretrained VAE in the previous part, and run the command, you will get images named end with afterVAE in that path.

```
cd test
python validate_pretrained_VAR.py --image_path="PATH_OF_IMAGES" --vae_path="PATH_TO_VAE"
```

## Acknowledgement

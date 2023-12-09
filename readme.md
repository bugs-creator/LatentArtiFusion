# Latent Artifusion

This is a latent space medical image restoration implementation based on https://github.com/zhenqi-he/ArtiFusion Engineering. The project is trained and tested on the diffusers framework. Based on https://arxiv.org/abs/2307.14262, we implemented a unique generated pipeline in `pipeline_latent_diffusion.py`.


## Environment Setup

First, make sure you have torch 2+ and cuda included in your environment, then install the environment:

```
pip install -r requirements.txt
```

Then generate the default config of training:

```
accelerate config default
```

Then download the pretrained VAE from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/vae. Put the file in a directory as below:

```
vae
├─config.json
├─diffusion_pytorch_model.bin
├─diffusion_pytorch_model.fp16.bin
├─diffusion_pytorch_model.fp16.safetensors
└─diffusion_pytorch_model.safetensors
```

## Use Our Pretrained Model


You can download out pretrained unet model from https://drive.google.com/file/d/1actPH17G3ksi051_hsGTIVSJeqTL0BbH/view?usp=sharing, then unzip it and put it in a folder.

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

## Train Your Model

You can train a new model from scratch. We provide two training codes for unet. Before the training begins, you need to complete the steps of the Environment Setup.

You can download our dataset from https://drive.google.com/drive/folders/13SDZzcgtO3RIdZIiteb-jo3hUtGAqOuq. Our model is trained on `Training_data/trainB`. You can also generate your own dataset by putting images in a directory.

Here is the launch script of training.

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

If you decide continue your last train, you can add the following args:

```
--resume_from_checkpoint=<checkpoint dir> 
```

We also provide a training script on another kind of unet. But it's less effective on latent space. Try it if you're interested.

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
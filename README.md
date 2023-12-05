The following libraries are necessary to run the program

conda create --name diffusion-huggingface python=3.10.12

conda activate diffusion-huggingface 

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install diffusers[training]

RUN PROGRAM

If we have only one GPU, it is necessary to modify the following parameters:

train_batch_size
eval_batch_size
image_size


@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 4 #*****************************
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

Since we change the batchsize we need to change the output dimension in evaluate
It is necessary to modify the image_grid to 2x2
  
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images
    print("images")
    print(images)
    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=2, cols=2)

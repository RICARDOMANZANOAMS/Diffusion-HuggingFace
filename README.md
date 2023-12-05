The following libraries are necessary to run the program

conda create --name diffusion-huggingface python=3.10.12

conda activate diffusion-huggingface 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install diffusers[training]

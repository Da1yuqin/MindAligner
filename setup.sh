#!/bin/bash

ENV_NAME="MindAligner"
PYTHON_VERSION="3.11"

conda create --name $ENV_NAME python=$PYTHON_VERSION -y
source activate $ENV_NAME
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install accelerate==0.24.1 \
            deepspeed==0.13.1 \
            diffusers==0.23.0 \
            einops \
            evaluate==0.4.1 \
            ftfy \
            h5py==3.10.0 \
            jupyter \
            jupyterlab \
            jupyterlab_nvdashboard \
            kornia==0.7.1 \
            matplotlib==3.8.2 \
            nltk==3.8.1 \
            numpy \
            omegaconf==2.3.0 \
            open_clip_torch \
            pandas==2.2.0 \
            pytorch-lightning==2.0.1 \
            regex \
            rouge_score==0.1.2 \
            scikit-image==0.22.0 \
            sentence-transformers==2.5.1 \
            torch==2.1.0 \
            torchmetrics==1.3.0.post0 \
            torchvision==0.16.0 \
            tqdm \
            transformers==4.37.2 \
            umap==0.1.1 \
            wandb \
            webdataset==0.2.73 \
            xformers==0.0.22.post7
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install dalle2-pytorch

echo "Conda environment '$ENV_NAME' with Python $PYTHON_VERSION has been created and activated."

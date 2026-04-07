# Select base image from https://cloud.ru/docs/aicloud/mlspace/concepts/environments__basic-images-list__jobs.html
FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py38:0.0.36

# Specifies that all subsequent commands must be executed as the root user
USER root

# Update the list of packages in the system and already installed packages
RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y software-properties-common && apt-get update

WORKDIR /home/user

# Install common ML/DL dependencies
RUN pip3 install --no-cache-dir \
    lightning-utilities \
    huggingface_hub \
    gymnasium \
    matplotlib \
    wandb \
    numpy \
    rich \
    h5py \
    tqdm \
    imageio \
    imageio-ffmpeg \
    pyrallis \
    omegaconf \
    stable-baselines3

# Copy patch file to container
COPY metadrive.patch /home/user/metadrive.patch
# Clone and install MetaDrive in one step (better layer caching)
RUN git clone https://github.com/metadriverse/metadrive.git && \
    cd metadrive && \
    git checkout 85e5dadc && \
    git apply /home/user/metadrive.patch && \
    pip install -e .

RUN cd /home/user/metadrive && \
    python -m metadrive.pull_asset --update

RUN pip3 install --no-cache-dir --no-build-isolation \
    flash-attn

COPY domain ./domain
COPY pyproject.toml .
RUN ls
RUN pip3 install -e .

# Change user to 'user'
USER user
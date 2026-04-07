FROM cr.ai.cloud.ru/aicloud-base-images/cuda11.8-torch2-py310:0.0.36

# Specifies that all subsequent commands must be executed as the root user
USER root

RUN apt-get update --fix-missing && apt-get upgrade -y &&\
 apt-get install -qy\
 python3-dev\
 build-essential\
 libblas-dev\
 libatlas-base-dev\
 libosmesa6-dev\
 libgl1-mesa-glx\
 libglfw3\
 patchelf\
 libgmp-dev\
 libgmpxx4ldbl\
 libglew-dev

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

RUN pip3 install "Cython<3" && pip3 install "humenv[bench] @ git+https://github.com/facebookresearch/HumEnv.git" && pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    stable_baselines3 \
    setuptools \
    packaging \
    wheel \
    pydantic && pip3 install --no-cache-dir \
    jupyter \
    lightning-utilities \
    huggingface_hub \
    pre-commit \
    gymnasium \
    matplotlib==3.10.1 \
    wandb \
    numpy==2.2.6 \
    rich \
    h5py \
    tqdm \
    imageio \
    imageio-ffmpeg \
    pyrallis \
    mujoco \
    mediapy \
    dm_control \
    pytest \
    scipy \
    paramiko \
    contourpy==1.3.2 \
    pandas==2.2.3 \
    omegaconf && git config --global http.postBuffer 1048576000 && \
    git config --global https.postBuffer 1048576000 && \
    git config --global http.version HTTP/1.1

WORKDIR /home/user

RUN pip install --upgrade pip
RUN pip install "Cython<3"
RUN pip uninstall --yes torch
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic

RUN pip install --no-cache-dir --no-build-isolation --use-pep517 flash-attn==2.7.0.post2

COPY domain ./domain
COPY pyproject.toml .
RUN ls
RUN pip3 install -e .
ENV BASE_HUMENV_PATH=/home/user/domain

USER user

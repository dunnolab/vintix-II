# Select base image from https://cloud.ru/docs/aicloud/mlspace/concepts/environments__basic-images-list__jobs.html
FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py38:0.0.36

### Set the working directory
WORKDIR /home/user/domains/controlgym

### Specifies that all subsequent commands must be executed as the root user
USER root

# Install torch and flashattn
RUN pip3 install "Cython<3"
RUN pip3 uninstall --yes torch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic

### Install dependencies
RUN pip install --no-cache-dir --no-build-isolation \
    gymnasium \
    flash-attn

### Copy patch files
COPY ./patches/ /home/user/patches/

### Install controlgym
RUN mkdir -p /home/user/site-packages/controlgym && \
    cd /home/user/site-packages/controlgym && \
    git clone https://github.com/xiangyuan-zhang/controlgym.git && \
    cd controlgym && \
    git apply /home/user/patches/deprecation-warning--forgotten-clipping--reset-kwargs.patch && \
    git apply /home/user/patches/extend-controller-usage.patch && \
    git apply /home/user/patches/more-controllers.patch && \
    git apply /home/user/patches/numpy-to-torch.patch && \
    git apply /home/user/patches/better-typing.patch && \
    python -m pip install -e .

COPY domain ./domain
COPY pyproject.toml .
RUN ls
RUN pip3 install -e .

# Change user to 'user'
USER user
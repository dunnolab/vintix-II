# Select base image from https://cloud.ru/docs/aicloud/mlspace/concepts/environments__basic-images-list__jobs.html
FROM cr.ai.cloud.ru/aicloud-base-images/cuda11.8-torch2-py39:0.0.36

# -------------------------------- ENERGYPLUS -------------------------------- #

# VERSION ARGUMENTS
ARG ENERGYPLUS_VERSION=24.1.0
ARG ENERGYPLUS_INSTALL_VERSION=24-1-0
ARG ENERGYPLUS_SHA=9d7789a3ac

ENV ENERGYPLUS_TAG=v$ENERGYPLUS_VERSION
ENV EPLUS_PATH=/usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION
# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-Ubuntu22.04-x86_64.sh
# Python add pyenergyplus path in order to detect API package
ENV PYTHONPATH="/usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}:/home/jovyan"

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

RUN wget https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64.sh \
    && chmod +x EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64.sh \
    && echo "y\r" | ./EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64.sh \
    && rm EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64.sh \
    && cd /usr/local/EnergyPlus-24-1-0 \
    && rm -rf PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater

#     Remove the broken symlinks
RUN cd /usr/local/bin && find -L . -type l -delete

COPY requirements.txt /home/user/

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

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

RUN pip install -r /home/user/requirements.txt

COPY domain ./domain
COPY pyproject.toml .
RUN ls
RUN pip3 install -e .

USER user
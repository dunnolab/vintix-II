FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36

USER root

RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y software-properties-common curl libpython3.8-dev

# Update pip
RUN pip install --upgrade pip

# Installing torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# Flash-attn
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
RUN pip install flash-attn==2.5.6 --no-build-isolation --no-cache-dir \
    -f https://github.com/Dao-AILab/flash-attention/releases
RUN python -c "from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_with_kvcache"

# Installing other dependencies
RUN pip install --no-cache-dir \
    "numpy>=1.24.4,<2.0.0" \
    pytest \
    gymnasium \
    matplotlib \
    wandb \
    h5py \
    tqdm \
    pyrallis

# Installing citylearn
RUN pip uninstall --yes scikit-learn
RUN pip install "citylearn==2.3.0"

# Installing citylearn datasets
WORKDIR /home/user
RUN git clone --depth 1 https://github.com/intelligent-environments-lab/CityLearn.git
RUN cp -r CityLearn/data/datasets ./
RUN rm -rf CityLearn

# Hardcode
RUN pip uninstall --yes scikit-learn
RUN pip install "scikit-learn>=1.3.0"

COPY domain ./domain
COPY datasets ./datasets
COPY pyproject.toml .
RUN ls
RUN pip3 install -e .

USER user
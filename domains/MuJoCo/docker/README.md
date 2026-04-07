# Python environment for MuJoCo

## Building docker image
Run the following command to build the image:

```shell
docker build --platform linux/amd64 -t mujoco-image:version1 .
```

## Manual installation
For model inference, the `numpy>=1.24.4`, `torch`, and `flash-attn` packages are required. However, ensure that `flash-attn` is compatible with the installed `torch` and CUDA version. For example, for CUDA 11.8, you can use the following command:
```shell
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip3 install --no-cache-dir --no-build-isolation \
    ninja \
    setuptools \
    packaging \
    pydantic \
    wheel
pip3 install --no-cache-dir --no-build-isolation flash-attn
```

`gymnasium>=0.28.0`, `mujoco<3` and `imageio` are required to create MuJoCo tasks.

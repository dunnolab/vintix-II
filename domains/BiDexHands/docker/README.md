# Python environment for Bi-DexHands

## Building docker image
Run the following command to build the image:

```shell
docker build --platform linux/amd64 -t bidexhands-image:version1 .
```

## Initializing Bi-DexHands Gym environment

In a newly created environment, run these commands:
```python
from envs.wrappers.bidexhands import get_gym_env, TASKS

print(f"Available tasks are:", *TASKS, sep='\n')

# Initialize env
env = get_gym_env(task_name="ShadowHandCatchUnderarm")

# Don't forget to call `.close()` before initializing a new env!
env.close()
```

## Notes

This Docker image supports only CPU-based environments due to the lack of
Vulkan installation.

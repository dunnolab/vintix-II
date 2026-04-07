import torch
from domain.environments import get_env_names, get_env

from vintix import Vintix2

if __name__ == '__main__':
    PATH_TO_CHECKPOINT = "path/to/checkpoint"
    model = Vintix2()
    model.load_model(PATH_TO_CHECKPOINT)
    model.to(torch.device('cuda'))
    model.eval()

    task_names = get_env_names()
    task_name = task_names[0]
    env = get_env(task_name)
    model.reset_context(task_name,
                        torch_dtype=torch.float16)
    max_episodes = 10
    episode_rewards = []
    print("START EVALUATING")

    for episode in range(max_episodes):
        cur_ep_rews = []
        observation, info = env.reset()
        reward = None
        done = False
        while not done:
            action = model.get_next_action(observation=observation,
                                           prev_reward=reward)
            observation, reward, termined, truncated, info = env.step(action)

            done = termined or truncated
            cur_ep_rews.append(reward)
        episode_rewards.append(sum(cur_ep_rews))
    print(f"Rewards per episode for {task_name}: {episode_rewards}")

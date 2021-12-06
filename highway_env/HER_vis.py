import gym
import highway_env
import numpy as np

from stable_baselines3 import SAC

env = gym.make("parking-v0")

model = SAC.load('her_sac_highway', env=env)

obs = env.reset()

episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs = env.reset()
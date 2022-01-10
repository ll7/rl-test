import gym

from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

tmp_path = "./tmp/CartPole_PPO"

new_logger = configure(tmp_path, ["tensorboard", "stdout"])

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tmp_path)
model.learn(total_timesteps=100_000)
model.save("./tmp/CartPole_PPO_model")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    sleep(0.05)
    if done:
      obs = env.reset()
import gym

from time import sleep

from stable_baselines3 import DQN

tmp_path = "./tmp/CartPole_DQN"

from stable_baselines3.common.logger import configure

new_logger = configure(tmp_path, ["tensorboard", "stdout"])

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1)

model.set_logger(new_logger)

model.learn(int(2e5))

model.save("./tmp/CartPole_DQN_model")


obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    sleep(0.05)
    if done:
      obs = env.reset()
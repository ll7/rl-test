import gym
import logging
from time import sleep
from stable_baselines3 import A2C

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

env = gym.make("CartPole-v1")


model = A2C.load("a2c_cartpole")

obs = env.reset()

# this is only visualization of the final result
for i in range(int(1e4)):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    logging.debug(obs)
    sleep(0.05)
    if done:
        logging.info('done: reset')
        env.reset()

env.close()

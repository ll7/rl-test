import os # to log to specific positions
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy # average reward over episodes. plus std deviation
from datetime import datetime

environment_name = 'CartPole-v0'
env = gym.make(environment_name)

episodes = 5
for episode in range(1, episodes +1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

log_path = os.path.join(os.getcwd(), 'logs')

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20_000)

PPO_Path = os.path.join(os.getcwd(), 'models', 'PPO_model_' + environment_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

model.save(PPO_Path)
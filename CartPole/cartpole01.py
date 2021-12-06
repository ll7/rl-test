import gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1")

model = A2C('MlpPolicy', env, verbose=2, tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=100_000) 
model.save("a2c_cartpole")

obs = env.reset()

# this is only visualization of the final result
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs.reset()
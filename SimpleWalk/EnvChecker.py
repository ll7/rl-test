import logging
from SimpleWalk2D import SimpleWalk2DDynGoal

env = SimpleWalk2DDynGoal()
episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    logging.debug('state: {}'.format(state))
    
    while not done:
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        logging.debug('Steps taken: {}'.format(info['steps_taken']))
        logging.debug('distance to goal: {}'.format(info['distance_to_goal']))
        score+=reward
    logging.info('Episode:{}'.format(episode)) #, score))
    logging.info('Score: {}'.format(score))
    env.render()
env.close()
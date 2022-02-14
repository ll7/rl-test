from gym import Env
from gym.spaces import Box
import numpy as np

import math

import matplotlib.pyplot as plt

# logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True # disable font warnings based on https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python



class SimpleWalk2DDynGoal(Env):
    """simple walk environment in 2D with a continuous action and state space"""
    def __init__(self):
        """Initialize the environment
        Big square as the environment and sub squares as the goal"""
        
        # set the dimensions of the environment
        # environment is square, therefore x and y are the same
        self.x_min = 0.0
        self.x_max = 20.0
        self.width = self.x_max - self.x_min
        
        
        # set maximum number of steps to reach the goal
        # if the maximum number of steps is reached, the episode is over
        # maximum is max steps to cross the field
        self.max_steps = int(math.ceil(self.x_max - self.x_min))
        
        # set goal boarder
        self.goal_boarder = 3.0
        
        # ensure that the goal can not walk out of the environment
        self.goal_max_speed =  self.goal_boarder / self.max_steps * 0.9
        
        self.viable_goal_distance = 0.5
        
        
        # set spaces
        self.action_space = Box(low=-1.0, high=1.0, shape=(2, )) # x and y change of position
        self.observation_space = Box(low=self.x_min, high=self.x_max, shape=(4, )) # x,y position, x,y goal
        
        # safe past states in an array, safe x and y positions
        self.state = np.ndarray(shape=(4,), dtype=np.float32)
        
        
    def __out_of_bounds(self):
        """check if the current state is out of bounds"""
        position = self.state[0:2]
        for element in position:
            if not (self.x_min <= element <= self.x_max):
                
                return True
            else:
                return False
            
    def __append_state(self):
        self.state_array[0].append(self.state[0])
        self.state_array[1].append(self.state[1])
        
    def __distance_to_goal(self):
        distance_to_goal = np.linalg.norm(self.state[0:2] - self.state[2:4])
        return distance_to_goal     
        
    
    def step(self, action):
        previous_state = self.state
        # update position
        self.state[0] += action[0] # update x
        self.state[1] += action[1] # update y
        self.__append_state()
        new_state = self.state
        self.steps_taken += 1
        distance_to_goal = self.__distance_to_goal()
    
        if self.steps_taken >= self.max_steps:
            # maximum number of steps reached
            logging.debug("maximum number of steps reached")
            reward = -100.0
            done = True
        elif self.__out_of_bounds():
            # went out of bounds
            logging.debug("out of bounds")
            reward = -100.0
            done = True
        elif distance_to_goal < self.viable_goal_distance:
            # reached goal
            reward = 1000.0
            logging.debug("reached goal")
            done = True
        elif False: # self.distance_to_goal > distance_to_goal:
            # got closer
            logging.debug("got closer")
            reward = (self.distance_to_goal - distance_to_goal) * 20
            done = False
        else:
            
            movement = self.distance_to_goal - distance_to_goal
            logging.debug("movement: {}".format(movement))
            reward = movement * 50
            done = False
        
        # update distance to goal
        self.distance_to_goal = distance_to_goal
        info = {'distance_to_goal': self.distance_to_goal, 
                'steps_taken': self.steps_taken,
                'previous_state': previous_state,
                'new_state': new_state}
        
        return self.state, reward, done, info
    
    def reset(self):
        """reset and initialize the environment"""
        
        # set x and y of position
        self.state[0:2] = np.random.uniform(low=self.x_min, high=self.x_max, size=(2,))
        
        # set goal to a random value, shrinked from entire environment size
        self.state[2:4] = np.random.uniform(
            low = self.x_min + self.goal_boarder, 
            high = self.x_max - self.goal_boarder,
            size=(2,)
            )
        
        self.state_array = [[], []] # x, y
        self.__append_state()
        self.steps_taken = 0
        self.distance_to_goal = self.__distance_to_goal()
        return self.state
    
    def render(self):
        
        # logging.debug("visited states: ", self.state_array)
        logging.debug("x: {}".format(self.state_array[0]))
        logging.debug("y: {}".format(self.state_array[1]))
        goal = self.state[2:4]
        logging.debug("goal: {}".format(goal))
        
        
        # plot
        fig, ax = plt.subplots()

        # plot the visited states
        ax.plot(self.state_array[0], self.state_array[1], linewidth=2.0)
        
        # plot the goal
        ax.plot(goal[0], goal[1], 'ro')
        circle1 = plt.Circle((goal[0], goal[1]), 1, color='r', fill=False)
        ax.add_patch(circle1)
        ax.set(
            xlim=(self.x_min, self.x_max), #xticks=np.arange(1, 8),
            ylim=(self.x_min, self.x_max), #yticks=np.arange(1, 8))
            )
        plt.show()

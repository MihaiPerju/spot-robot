import random
import time
import mujoco
from collections import deque
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from IPython.display import display, clear_output
import wandb
import os

class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu, theta, sigma):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def generate(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        return self.state

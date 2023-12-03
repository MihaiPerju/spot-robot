import random
import numpy as np
from collections import deque
import tensorflow as tf
from collections import namedtuple
import wandb
import os

Experience = namedtuple(
    "Experience", ["observation", "action", "reward", "next_observation", "done"])

# AGENT


class Agent():
    def __init__(
        self,

        env,
        policy,

        n_episodes,

        checkpoint_episode_interval=100,
        steps_before_learning=20,

        experience_replay_size=10000,
        batch_size=128,

    ):
        self.env = env
        self.policy = policy
        self.n_episodes = n_episodes
        self.steps_before_learning = steps_before_learning
        self.checkpoint_episode_interval = checkpoint_episode_interval
        self.last_checkpoint = 0

        self.experience_replay = deque(maxlen=experience_replay_size)
        self.batch_size = batch_size
        self.rewards = deque(maxlen=100)

        self.n_steps = 0
        self.total_steps = 0
        self.max_reward = -1
        self.episode = 1

    def log(self, message):
        print(message)

    def sample_batch(self, batch_size):
        experiences = random.sample(self.experience_replay, batch_size)

        # Extract the fields from the experiences
        observations, actions, rewards, next_observations, dones = zip(
            *experiences)

        # Convert each list into a batched tensor
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_observations = tf.convert_to_tensor(
            next_observations, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        return {"observations": observations, "actions": actions, "rewards": rewards, "next_observations": next_observations, "dones": dones}

    def get_avg_reward(self):
        return np.mean(self.rewards)

    def train(self):
        for episode in range(self.n_episodes):
            self.episode = episode

            obs = self.env.reset()
            done = False
            while not done:
                self.total_steps += 1
                wandb.log({"Total steps": self.total_steps})

                action = self.policy.activate(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                wandb.log({"Average reward": self.get_avg_reward()})

                if reward > self.max_reward:
                    self.max_reward = reward
                    wandb.log({"Max Reward": reward})

                wandb.log({"Reward": reward})
                experience = Experience(obs, action, reward, next_obs, done)
                obs = next_obs

    def interact(self):
        self.log(f"Interacting with the environment")

        obs = self.env.reset()
        self.n_steps = 0
        done = False
        while not done:
            self.n_steps += 1
            action = self.policy.activate(obs)
            next_obs, reward, done, _ = self.env.step(action)
            obs = next_obs

        self.env.recorder.show_video()

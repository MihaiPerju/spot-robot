import random
from collections import deque
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
        steps_before_learning=20,

        experience_replay_size=10000,
        batch_size=128,
    ):
        self.env = env
        self.policy = policy
        self.n_episodes = n_episodes
        self.steps_before_learning = steps_before_learning

        self.experience_replay = deque(maxlen=experience_replay_size)
        self.batch_size = batch_size
        self.rewards = deque(maxlen=100)

        self.n_steps = 0
        self.total_steps = 0
        self.max_reward = -1
        self.episode = 1

    def update_reward_graph(self):
        # clear_output(wait=True)
        os.system('clear')

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

    def learn(self, experience):
        self.experience_replay.append(experience)

        if (self.total_steps % self.steps_before_learning == 0) and (len(self.experience_replay) > self.batch_size):
            self.log("LEARN")
            batch = self.sample_batch(self.batch_size)
            self.policy.update(batch)

    def train(self):
        for episode in range(self.n_episodes):
            self.episode = episode
            wandb.log({"Episode": episode})

            obs = self.env.reset()
            done = False
            while not done:

                self.total_steps += 1

                action = self.policy.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)

                if reward > self.max_reward:
                    self.max_reward = reward
                    wandb.log({"Max Reward": reward})

                # adding the reward and plot the new rewards axis
                # self.rewards.append(reward)
                self.update_reward_graph()
                wandb.log({"Reward": reward})

                experience = Experience(obs, action, reward, next_obs, done)
                self.learn(experience)

                obs = next_obs

                # enabling or disabling noise
                if self.total_steps % 1000 == 0:
                    reached_distance_variance = self.env.get_reached_distances_variance()
                    print(f"REACHED DIST VARIANCE: {reached_distance_variance}")
                    wandb.log({"Reached distance variance": reached_distance_variance})
                    if reached_distance_variance < 0.05 and self.policy.noise_state == 0:
                        self.policy.set_noise(1)
                    if reached_distance_variance >= 0.05 and self.policy.noise_state == 1:
                        self.policy.set_noise(0)

            # cleaning the memory and saving the weights
            if self.episode % 1000 == 0 or self.total_steps % 10000 == 0:
                self.policy.save_weights()
                tf.keras.backend.clear_session()

    def interact(self):
        self.log(f"Interacting with the environment")

        obs = self.env.reset()
        self.n_steps = 0
        done = False
        while not done:
            self.n_steps += 1
            action = self.policy.select_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            obs = next_obs

        self.env.recorder.show_video()

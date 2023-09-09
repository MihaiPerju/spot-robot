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


# opening the scene
xml_file_path = './scene.xml'

with open(xml_file_path, 'r') as file:
    xml = file.read()


def get_standing_spot_model():
  spot = mujoco.MjModel.from_xml_string(xml)
  spot_data = mujoco.MjData(spot)
  keyframe = spot.keyframe("home")

  spot_data.qpos=keyframe.qpos
  spot_data.ctrl = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8,]

  return spot, spot_data

wandb.login()

# ENVIRONMENT
class Recorder():
  def __init__(self, model, framerate=24):
    self.model = model
    self.data = mujoco.MjData(self.model)
    self.frames=[]
    self.framerate = framerate

  def reset(self):
    pass

class SpotEnvironment(gym.Env):
    def __init__(self, goal_distance=5, steps_per_episode=1000, should_render=False,):
        self.model, self.data = get_standing_spot_model()
        self.action_delay=0.2
        self.should_render=should_render
        self.goal_distance=goal_distance
        self.max_distance=-1

        self.n_steps=0
        self.steps_per_episode=steps_per_episode

        if should_render:
          self.recorder = Recorder(self.model)

        # Set random seed
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.model, self.data = get_standing_spot_model()

        if self.should_render:
          self.recorder.reset()

        observation = self.get_observation()
        return observation

    def get_z_axis_rotation(self):

      def quaternion_to_rotation_matrix(q):
          w, x, y, z = q
          return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                          [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                          [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])

      quaternion = self.data.body("trunk").xquat
      rotation_matrix = quaternion_to_rotation_matrix(quaternion)

      # Step 2: Extract the local z-axis (up vector) from the rotation matrix
      local_z_axis = rotation_matrix[:, 2]

      # Step 3: Check if the object is upside down (opposite direction of the world's z-axis)
      result = local_z_axis[2] < 0
      # print(f"Z axis {local_z_axis[2]}")

      return local_z_axis[2]

    def get_observation(self):
      bodies=[
        'trunk',

        'FL_calf',
        'FL_thigh',

        'FR_calf',
        'FR_thigh',

        'RL_calf',
        'RL_thigh',

        'RR_calf',
        'RR_thigh',
      ]
      observation = []

      # reached_distance = self.data.body("trunk").xpos[0]
      # distance_to_goal = self.goal_distance - reached_distance

      trunk_rotation = self.get_z_axis_rotation()
      trunk_height = self.data.body("trunk").xpos[2]
      trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

      observation.append([
          trunk_rotation,
      ])

      for body in bodies:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        observation.append([
            *self.data.xpos[body_id],
            self.data.qpos[body_id],
            self.data.qvel[body_id],
        ])

      # for site in ["FL","FR","RL","RR"]:
      #   site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
      #   foot_location = self.data.site_xpos[site_id]
      #   observation.append(foot_location)

      return np.concatenate(observation)

    def rescale_action(self, raw_action):
      low_bound = np.array([0, -0.686, -2.818]*4)
      high_bound = np.array([0, 4.501, -0.888]*4)

      action_range = high_bound - low_bound
      rescaled_action = low_bound + ((raw_action + 1.0) * 0.5) * action_range

      return rescaled_action

    def reshape(self, raw_action):
      expanded_action = np.array([
          0, *raw_action[[0,1]],
          0, *raw_action[[2,3]],
          0, *raw_action[[4,5]],
          0, *raw_action[[6,7]],
      ])

      action = self.rescale_action(expanded_action)
      return action

    def step(self, raw_action):
        self.n_steps +=1

        action=self.reshape(raw_action)
        self.data.ctrl=action

        update_time=self.data.time+self.action_delay
        while self.data.time<update_time:
          mujoco.mj_step(self.model, self.data)
          self.render()

        done=False
        if self.n_steps>=self.steps_per_episode:
          done=True
          self.n_steps=0

        reached_distance = self.data.body("trunk").xpos[0]
        z_axis_rotation=self.get_z_axis_rotation()

        # the default reward is based on how far it got compared to destination
        # the closer to the destination - the more the reward will increase
        reward = reached_distance/self.goal_distance

        # if the robot reached the destination - it won
        if reached_distance>self.goal_distance:
          reward*=10

        elif reached_distance<0:
          reward*=1.5
        elif z_axis_rotation<0:
            reward*=2
        else:
            reward*=7

        info={}
        observation = self.get_observation()
        return observation, reward, done, info

    def render(self):
      if not self.should_render:
        return

      if len(self.recorder.frames) < self.data.time * self.recorder.framerate:
        self.recorder.add_frame(self.model, self.data)


Experience = namedtuple("Experience", ["observation", "action", "reward", "next_observation", "done"])

# AGENT
class Agent():
  def __init__(
      self,

      env,
      policy,

      n_episodes=20,
      steps_before_learning=20,

      experience_replay_size=10000,
      batch_size=128,
  ):
    self.env = env
    self.policy = policy
    self.n_episodes = n_episodes
    self.steps_before_learning=steps_before_learning

    self.experience_replay = deque(maxlen=experience_replay_size)
    self.batch_size = batch_size
    self.rewards = deque(maxlen=100)

    self.n_steps = 0
    self.total_steps=0
    self.max_reward=-1
    self.episode=1

  def update_reward_graph(self):
    # clear_output(wait=True)
    os.system('clear')


  def log(self, message):
    print(message)

  def sample_batch(self, batch_size):
    experiences = random.sample(self.experience_replay, batch_size)

    # Extract the fields from the experiences
    observations, actions, rewards, next_observations, dones = zip(*experiences)

    # Convert each list into a batched tensor
    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_observations = tf.convert_to_tensor(next_observations, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    return {"observations": observations, "actions": actions, "rewards": rewards, "next_observations": next_observations, "dones": dones}

  def learn(self, experience):
    self.experience_replay.append(experience)

    if (self.total_steps%self.steps_before_learning==0) and (len(self.experience_replay)>self.batch_size):
      self.log("LEARN")
      batch = self.sample_batch(self.batch_size)
      self.policy.update(batch)

  def train(self):
    for episode in range(self.n_episodes):
      self.episode=episode
      wandb.log({"Episode": episode})

      # every 100 episode - save weights
      if self.episode%100:
        self.policy.save_weights()

      obs = self.env.reset()
      done=False
      while not done:

        self.total_steps+=1

        action = self.policy.select_action(obs)
        next_obs, reward, done, _ = self.env.step(action)

        if reward>self.max_reward:
          self.max_reward = reward
          wandb.log({"Max Reward": reward})

        # adding the reward and plot the new rewards axis
        # self.rewards.append(reward)
        self.update_reward_graph()
        wandb.log({"Reward": reward})

        experience = Experience(obs, action, reward, next_obs, done)
        self.learn(experience)

        obs=next_obs

  def interact(self):
      self.log(f"Interacting with the environment")

      obs = self.env.reset()
      self.n_steps=0
      done=False
      while not done:
        self.n_steps+=1
        action = self.policy.select_action(obs)
        next_obs, reward, done, _ = self.env.step(action)
        obs=next_obs

      self.env.recorder.show_video()


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

class DDPG():
  def __init__(self, state_shape, action_shape, ou_noise, num_layers=5, layer_size=10, tau=0.005, gamma=0.99):

    self.tau = tau
    self.gamma = gamma
    self.layer_size = layer_size
    self.num_layers = num_layers

    self.actor = self.build_actor_network(state_shape, action_shape, num_layers, layer_size )
    self.critic = self.build_critic_network(state_shape, action_shape, num_layers, layer_size )
    self.target_actor = self.build_actor_network(state_shape, action_shape, num_layers, layer_size )
    self.target_critic = self.build_critic_network(state_shape, action_shape, num_layers, layer_size )

    self.actor_optimizer = tf.keras.optimizers.Adam()
    self.critic_optimizer = tf.keras.optimizers.Adam()

    # setting the weights for target networks
    self.target_actor.set_weights(self.actor.get_weights())
    self.target_critic.set_weights(self.critic.get_weights())

    self.ou_noise=ou_noise

  def build_actor_network(self, in_shape, out_shape, num_layers, layer_size, ):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(layer_size, activation="relu", input_shape=(in_shape,)))

    for _ in range(num_layers - 1):
      model.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    model.add(tf.keras.layers.Dense(out_shape, activation="tanh"))

    return model

  def build_critic_network(self, state_shape, action_shape, num_layers, layer_size ):
    state_input = tf.keras.layers.Input(shape=state_shape)
    action_input = tf.keras.layers.Input(shape=action_shape)

    state_output = tf.keras.layers.Dense(layer_size, activation="relu")(state_input)
    action_output = tf.keras.layers.Dense(layer_size, activation="relu")(action_input)

    for _ in range(num_layers-1):
      state_output=tf.keras.layers.Dense(layer_size, activation="relu")(state_output)
      action_output=tf.keras.layers.Dense(layer_size, activation="relu")(action_output)

    merged = tf.keras.layers.Concatenate()([state_output, action_output])
    merged_out=tf.keras.layers.Dense(layer_size, activation="relu")(merged)

    output = tf.keras.layers.Dense(1)(merged_out)

    model = tf.keras.models.Model(inputs=[state_input, action_input], outputs = output)

    return model

  def select_action(self, observation):
      observation = np.array(observation)
      observation = np.expand_dims(observation, axis=0)  # adds an additional dimension

      raw_action = self.actor.predict(observation)[0]
      exploration_noise = self.ou_noise.generate()
      action = raw_action + exploration_noise

      return action

  def update(self, batch):
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_observations = batch["next_observations"]
    dones = batch["dones"]

    print("training critic")
    # training the critic
    with tf.GradientTape() as tape:
      q_values = self.critic([observations, actions])

      next_actions = self.target_actor(next_observations)
      target_q_values = self.target_critic([next_observations, next_actions])
      target_q_values = rewards+self.gamma*target_q_values*(1-dones)

      critic_loss = tf.keras.losses.MSE(target_q_values, q_values)

    critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
    self.critic_optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

    print("training actor")
    # training the actor
    with tf.GradientTape() as tape:
      actions = self.actor(observations)
      critic_value = self.critic([observations, actions])

      actor_loss = -tf.math.reduce_mean(critic_value)

    actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
    self.actor_optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

    # soft update the weights of the target actor/critic
    actor_weights  = self.actor.get_weights()
    target_actor_weights = self.target_actor.get_weights()
    for i in range(len(target_actor_weights)):
      target_actor_weights[i]= (self.tau*actor_weights[i]) + (1-self.tau)*target_actor_weights[i]

    self.target_actor.set_weights(target_actor_weights)

    critic_weights  = self.critic.get_weights()
    target_critic_weights = self.target_critic.get_weights()
    for i in range(len(target_critic_weights)):
      target_critic_weights[i]= (self.tau*critic_weights[i]) + (1-self.tau)*target_critic_weights[i]

    self.target_critic.set_weights(target_critic_weights)

  def save_weights(self):
    timestamp = time.strftime("%Y%m%d%H%M%S")
    actor_filename = f"actor_{timestamp}.h5"
    target_actor_filename = f"target_actor_{timestamp}.h5"
    critic_filename = f"critic_{timestamp}.h5"
    target_critic_filename = f"target_critic_{timestamp}.h5"

    self.actor.save(actor_filename)
    self.target_actor.save(target_actor_filename)

    self.critic.save(critic_filename)
    self.target_critic.save(target_critic_filename)

    wandb.save(actor_filename)
    wandb.save(target_actor_filename)
    wandb.save(critic_filename)
    wandb.save(target_critic_filename)

  def load_weights(self, actor_model,target_actor_model,critic_model,target_critic_model,):
    self.actor.load_weights(actor_model.name)
    self.target_actor.load_weights(target_actor_model.name)
    self.critic.load_weights(critic_model.name)
    self.target_critic.load_weights(target_critic_model.name)


for num_layers in [7]:
  for layer_size in [60,70,80,90,100]:
    sample_env = SpotEnvironment()
    observation_sample = sample_env.get_observation()

    config=dict(
        ou = dict(
            size=8,
            mu=0,
            theta=0.8,
            sigma=0.5,
        ),
        # state and actions
        state_shape=observation_sample.shape[0],
        action_shape=8,

        # network configuration
        num_layers=num_layers,
        layer_size=layer_size,

        # training
        n_episodes=10,
        steps_before_learning=40,
        steps_per_episode=10000,
        goal_distance=5,
    )

    wandb.init(name=f"nn {num_layers}x{layer_size} {config['n_episodes']}ep x {config['steps_per_episode']}", project="spot-2", config=config)
    config = wandb.config

    ou_noise = OrnsteinUhlenbeckProcess(size=config.ou['size'], mu=config.ou['mu'], theta=config.ou['theta'], sigma=config.ou['sigma'])
    policy = DDPG(state_shape=config.state_shape, action_shape=config.action_shape, num_layers=config.num_layers, layer_size=config.layer_size, ou_noise=ou_noise)


    spot_env = SpotEnvironment(steps_per_episode=config.steps_per_episode, goal_distance=config.goal_distance)
    agent = Agent(env=spot_env, policy=policy, n_episodes=config.n_episodes, steps_before_learning=config.steps_before_learning,)

    agent.train()
    policy.save_weights()
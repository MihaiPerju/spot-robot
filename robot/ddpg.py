
import time
import numpy as np
import tensorflow as tf
import wandb

from noise import OrnsteinUhlenbeckProcess

class DDPG():
  def __init__(self, state_shape, action_shape, ou, num_layers=5, layer_size=10, tau=0.005, gamma=0.99):

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
    self.ou = ou
    self.ou_noise=OrnsteinUhlenbeckProcess(size=self.ou['size'], mu=self.ou['mu'], theta=self.ou['theta'], sigma=self.ou['sigma'])
    

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
      critic_loss_mean = np.mean(critic_loss)
      wandb.log({"Critic Loss": critic_loss_mean})

    critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
    self.critic_optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

    print("training actor")
    # training the actor
    with tf.GradientTape() as tape:
      actions = self.actor(observations)
      critic_value = self.critic([observations, actions])

      actor_loss = -tf.math.reduce_mean(critic_value)
      wandb.log({"Actor Loss": actor_loss})

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
    actor_filename = f"models/actor_{timestamp}.h5"
    target_actor_filename = f"models/target_actor_{timestamp}.h5"
    critic_filename = f"models/critic_{timestamp}.h5"
    target_critic_filename = f"models/target_critic_{timestamp}.h5"

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

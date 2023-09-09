import wandb

from ddpg import DDPG
from agent import Agent
from environment import SpotEnvironment

wandb.login()

for num_layers in [20]:
  for layer_size in [80, 100, 120, 150]:
    sample_env = SpotEnvironment(steps_per_episode=300, goal_distance=100)
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
        n_episodes=20,
        steps_before_learning=50,
        steps_per_episode=10000,
        goal_distance=15,
    )

    wandb.init(
      name=f"{num_layers}x{layer_size} neurons {config['n_episodes']} x {config['steps_per_episode']}steps",
      project="spot-3", 
      config=config, 
      reinit=True
    )
    
    config = wandb.config
    print(config)

    policy = DDPG(state_shape=config.state_shape, action_shape=config.action_shape, num_layers=config.num_layers, layer_size=config.layer_size, ou=config.ou)

    spot_env = SpotEnvironment(steps_per_episode=config.steps_per_episode, goal_distance=config.goal_distance)
    agent = Agent(env=spot_env, policy=policy, n_episodes=config.n_episodes, steps_before_learning=config.steps_before_learning,)

    agent.train()
    policy.save_weights()
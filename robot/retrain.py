import wandb

from ddpg import DDPG
from agent import Agent
# from environment_no_falls import SpotEnvironmentNoFalls
from environment_no_falls_v2 import SpotEnvironmentNoFalls

wandb.login()

# loading the weights
wandb_api = wandb.Api()

models =[
  ("20230909185842", "scg4ycc6"),
]

for model in models:
  timestamp=model[0]
  run_id=model[1]
  project_name = "spot-no-falls"
  sample_env = SpotEnvironmentNoFalls(steps_per_episode=300, goal_distance=100)
  observation_sample = sample_env.get_observation()


  run = wandb_api.run(f"mikeperju/{project_name}/{run_id}")
  run.config['n_episodes']=1000000
  run.config['steps_per_episode']=100000
  
  wandb.init(
    name=f"{run.config['num_layers']}x{run.config['layer_size']} neurons {run.config['n_episodes']} x {run.config['steps_per_episode']}steps",
    project="spot-no-falls", 
    config=run.config, 
    reinit=True

    )
  config = wandb.config

  # config.ou['theta']=0.8
  # config.ou['sigma']=0.5

  policy = DDPG(state_shape=config.state_shape, action_shape=config.action_shape, num_layers=config.num_layers, layer_size=config.layer_size, ou=config.ou)

  actor_model = wandb.restore(f'actor_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")
  target_actor_model = wandb.restore(f'target_actor_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")
  critic_model = wandb.restore(f'critic_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")
  target_critic_model = wandb.restore(f'target_critic_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")

  policy.load_weights(
      actor_model,
      target_actor_model,
      critic_model,
      target_critic_model,
  )
  spot_env = SpotEnvironmentNoFalls(steps_per_episode=config.steps_per_episode, goal_distance=config.goal_distance)
  agent = Agent(env=spot_env, policy=policy, n_episodes=config.n_episodes, steps_before_learning=config.steps_before_learning,)

  agent.train()
  policy.save_weights()


# PID 90272
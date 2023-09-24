import wandb

from ddpg import DDPG
from agent import Agent
from environment_lr import SpotEnvironment

wandb.login()

# loading the weights
wandb_api = wandb.Api()

models =[
  ("20230923043356", "6hc3dbz0"),
  ("20230923031906", "ohugu5yb"),
]

for model in models:
  timestamp=model[0]
  run_id=model[1]
  project_name = "spot-learning-rate"
  sample_env = SpotEnvironment(steps_per_episode=300, goal_distance=100)
  observation_sample = sample_env.get_observation()

  run = wandb_api.run(f"mikeperju/{project_name}/{run_id}")
  run.config['n_episodes']=100000
  run.config['steps_per_episode']=100000
  run.config['ou']['theta']=0.8
  run.config['ou']['sigma']=0.5

  wandb.init(
    name=f"{run.config['num_layers']}x{run.config['layer_size']} neurons {run.config['n_episodes']} x {run.config['steps_per_episode']}steps",
    project = "spot-lr-noise-2",
    config=run.config, 
    reinit=True
    )
  config = wandb.config

  policy = DDPG(state_shape=config.state_shape, action_shape=config.action_shape, num_layers=config.num_layers, layer_size=config.layer_size, ou=config.ou)

  actor_model = wandb.restore(f'models/actor_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")
  target_actor_model = wandb.restore(f'models/target_actor_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")
  critic_model = wandb.restore(f'models/critic_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")
  target_critic_model = wandb.restore(f'models/target_critic_{timestamp}.h5', run_path=f"mikeperju/{project_name}/{run_id}")

  policy.load_weights(
      actor_model,
      target_actor_model,
      critic_model,
      target_critic_model,
  )
  spot_env = SpotEnvironment(steps_per_episode=config.steps_per_episode, goal_distance=config.goal_distance)
  agent = Agent(env=spot_env, policy=policy, n_episodes=config.n_episodes, steps_before_learning=config.steps_before_learning,)

  agent.train()
  policy.save_weights()

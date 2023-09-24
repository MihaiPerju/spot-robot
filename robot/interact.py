import wandb

from ddpg import DDPG
from agent import Agent
from environment_lr import SpotEnvironment

wandb.login()

# loading the weights
wandb_api = wandb.Api()

models =[
  ("20230924023209", "uw3mz9rw")
]

for model in models:
  timestamp=model[0]
  run_id=model[1]
  project_name = "spot-lr-noise"
  sample_env = SpotEnvironment(steps_per_episode=300, goal_distance=100)
  observation_sample = sample_env.get_observation()

  run = wandb_api.run(f"mikeperju/{project_name}/{run_id}")
  run.config['steps_per_episode']=1000
  
  wandb.init(
    name="visualisation",
    project="todelete", 
    config=run.config, 
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
  spot_env = SpotEnvironment(steps_per_episode=config.steps_per_episode, goal_distance=config.goal_distance, should_render=True)
  agent = Agent(env=spot_env, policy=policy, n_episodes=config.n_episodes, steps_before_learning=config.steps_before_learning,)

  max_distance = 0
  # while max_distance<4:
  agent.interact()
  reached_distance = spot_env.data.body("trunk").xpos[0]
  max_distance =  spot_env.max_distance

  print("Reached distance: ", reached_distance)
  print("Max distance: ", max_distance)



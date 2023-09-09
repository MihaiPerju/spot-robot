import wandb

from ddpg import DDPG
from agent import Agent
from environment_no_falls_v2 import SpotEnvironmentNoFalls

wandb.login()

# loading the weights
wandb_api = wandb.Api()

models =[
  ("20230909223542", "z591k1h3"),
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
    name="visualisation",
    project="spot-no-falls", 
    config=run.config, 
    )
  config = wandb.config

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
  spot_env = SpotEnvironmentNoFalls(steps_per_episode=config.steps_per_episode, goal_distance=config.goal_distance, should_render=True)
  agent = Agent(env=spot_env, policy=policy, n_episodes=config.n_episodes, steps_before_learning=config.steps_before_learning,)

  max_distance = 0
  while max_distance<4:
    agent.interact()
    reached_distance = spot_env.data.body("trunk").xpos[0]
    max_distance =  spot_env.max_distance

    print("Reached distance: ", reached_distance)
    print("Max distance: ", max_distance)



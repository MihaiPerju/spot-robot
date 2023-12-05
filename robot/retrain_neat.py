import neat
import wandb
import pickle
import platform
import os
from environment_neat import SpotEnvironment
from agent_neat import Agent
from datetime import datetime


config_path = "./neat-config.txt"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)
op_sys = platform.system()
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
current_date = datetime.now().strftime("%Y-%b-%d-%H:%M")

logs_dir = f'./neat-models/{op_sys}/{current_date}'
os.makedirs(logs_dir, exist_ok=True)


def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config)

        wandb.init(
            project="5-dec-1024",
            config=dict(
                genome_id=genome_id,
                config=str(config.genome_config)
            ),
            reinit=True
        )
        # Ensure SpotEnvironment is initialized correctly
        spot_env = SpotEnvironment(steps_per_episode=10000, goal_distance=10)

        agent = Agent(
            env=spot_env,
            policy=neural_network,
            n_episodes=1,
            steps_before_learning=50,
            checkpoint_episode_interval=101,
        )

        agent.train()

        # Consider using a more meaningful fitness calculation
        genome.fitness = agent.get_avg_reward()

def run_neat():

    checkpointer = neat.Checkpointer(
        generation_interval=1,
        time_interval_seconds=1800,  #  30 minutes
        filename_prefix=f'{logs_dir}/')

    checkpoint_path = "./neat-models/Darwin/2023-Dec-03-19:13/112"
    population = checkpointer.restore_checkpoint(checkpoint_path)

    # Run NEAT evolution
    best_genome = population.run(evaluate_genomes, 1000)
    # # Create a neural network based on the best genome
    # best_neural_network = neat.nn.FeedForwardNetwork.create(
    #     best_genome, config)

    # # Initialize SpotEnvironment with rendering enabled
    # spot_env = SpotEnvironment(
    #     steps_per_episode=10000, goal_distance=10, should_render=True)

    # # Initialize Agent with the winning policy
    # agent = Agent(
    #     env=spot_env,
    #     policy=best_neural_network,
    #     n_episodes=10000,
    #     steps_before_learning=50,
    #     checkpoint_episode_interval=101,
    # )

    # # Interact with the environment using the winning policy
    # agent.interact()

    # # Retrieve and print information about the interaction
    # reached_distance = spot_env.data.body("trunk").xpos[0]
    # max_distance = spot_env.max_distance

    # print("Reached distance:", reached_distance)
    # print("Max distance:", max_distance)


run_neat()

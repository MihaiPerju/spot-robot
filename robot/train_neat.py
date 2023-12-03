import neat
import wandb

from environment_neat import SpotEnvironment
from agent_neat import Agent


config_path = "./neat-config.txt"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

wandb.init(
    project="spot-neat",
    config=dict(),
    reinit=True
)


def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config)

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
        genome.fitness = agent.get_final_fitness()


def run_neat():

    population = neat.Population(config)
    reporter = neat.StdOutReporter(show_species_detail=True)
    population.add_reporter(reporter)
    population.add_reporter(neat.StatisticsReporter())

    # Run NEAT evolution
    winner = population.run(evaluate_genomes, 100)
    best_genome = winner
    print('\nBest genome:\n{!s}'.format(winner))

    # Create a neural network based on the best genome
    best_neural_network = neat.nn.FeedForwardNetwork.create(
        best_genome, config)

    # Initialize SpotEnvironment with rendering enabled
    spot_env = SpotEnvironment(
        steps_per_episode=10000, goal_distance=10, should_render=True)

    # Initialize Agent with the winning policy
    agent = Agent(
        env=spot_env,
        policy=best_neural_network,
        n_episodes=10000,
        steps_before_learning=50,
        checkpoint_episode_interval=101,
    )

    # Interact with the environment using the winning policy
    agent.interact()

    # Retrieve and print information about the interaction
    reached_distance = spot_env.data.body("trunk").xpos[0]
    max_distance = spot_env.max_distance

    print("Reached distance:", reached_distance)
    print("Max distance:", max_distance)


run_neat()

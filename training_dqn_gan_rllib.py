from env.pok_env_gym_RLLib import PokemonEnv
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from agent.DQNGAN import GAN_DQN
from gan.GANExploration import GANExploration
from ray.tune.logger import pretty_print
import ray
from ray.tune.registry import register_env

def env_creator(arg):
    print(arg)
    return PokemonEnv(render_mode="rgb_array")

register_env("PokemonEnv", env_creator)

ray.init()
input_size = (120,120)

for i in range(10):
    replay_config = {
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 60000,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.5,
            "prioritized_replay_eps": 3e-6,
        }

    config = (DQNConfig()
            .training(replay_buffer_config=replay_config,
                            train_batch_size=32,
                            n_step=1,
                            num_atoms=1,
                            double_q=False,
                            dueling=False,
                            num_steps_sampled_before_learning_starts=100)
                .resources(num_gpus=0)
                .rollouts(num_rollout_workers=1)
                .environment(env= "PokemonEnv")
                .exploration(exploration_config={
                    "type": GANExploration,
                    "file_path_for_gan_weights":"",
                    "input_size": input_size,
                }))
    algo = GAN_DQN(config=config)

    for _ in range(30):
        print(pretty_print(algo.train())) 
    
    print("Training done")
    # print("Saving model")
    # checkpoint = algo.save()
    # print(checkpoint)
    # print("Model saved")
ray.shutdown()



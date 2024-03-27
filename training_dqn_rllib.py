
from env.pok_env_gym_RLLib import PokemonEnv
from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
from ray.tune.logger import pretty_print
import ray
from ray.tune.registry import register_env
from time import time

def env_creator(arg):
    print(arg)
    return PokemonEnv(render_mode="human")



input_size = (120,120)
#Start time
start = time()

for i in range(1):
    ray.init()
    register_env("PokemonEnv", env_creator)

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
                .rollouts(num_rollout_workers=5)
                .environment(env= "PokemonEnv"))
    algo = DQN(config=config)

    for _ in range(1):
        print(pretty_print(algo.train())) 
    
    print("Training done")
    # print("Saving model")
    # checkpoint = algo.save()
    # print(checkpoint)
    # print("Model saved")
    del algo
    ray.shutdown()

#End time
end = time()

print("Time taken: ", end-start)



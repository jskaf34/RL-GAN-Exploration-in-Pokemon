from video_env import RedGymEnv
from pathlib import Path

ep_length = 2**23
sess_path = Path('session')

env_config = {
                'headless': False, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': 'jeu/init_state_pokeball.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'jeu/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }

# Usage example:
env = RedGymEnv(env_config)
observation = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs_memory, rew, done, step_limit_reached, dic  = env.step(action)

env.close()
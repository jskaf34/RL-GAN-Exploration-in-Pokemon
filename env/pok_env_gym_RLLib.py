import yaml
import os
import numpy as np
import pandas as pd
import mediapy as media
import gymnasium as gym

from pyboy import PyBoy, WindowEvent
from .memory_addresses import *
from .memory import ExplorationMemory
from gymnasium.spaces import Discrete, Box



class PokemonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    

    def __init__(self, config_path_from_dir_folder = "configs/env_config.yaml", render_mode='human') -> None:
        """
        Initialize the PokemonBlueEnv environment.

        Parameters:
        - rom_path (str): Path to the Pokemon Blue ROM file.
        - emulation_speed (int): Emulation speed for PyBoy.
        - start_level (int): Initial level for the player.
        - render_reward (bool): Whether to print reward details during rendering.

        Returns:
        - None
        """
        dir_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(dir_project, config_path_from_dir_folder)

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.observation_space = Box(low=0, high=255, shape=config["im_dim"],  dtype=np.uint8)
        self.action_space = Discrete(config["nb_action"]) # 6: no action

        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Invalid render_mode: {render_mode}"
        self.render_mode = render_mode
        if self.render_mode == "human" : 
            self.pyboy = PyBoy(config["rom_path"])
        else :
            self.pyboy = PyBoy(config["rom_path"], window_type="headless")
        self.pyboy.set_emulation_speed(config["emulation_speed"])

        self.init_state = config["init_state"]
        self.done = False
        self.resize_shape = (config["im_dim"][1], config["im_dim"][0])
        self.nb_step = 0
        self.max_step = config['ep_length']
        self.action_freq = config['action_freq']
        self.im_dim = config["im_dim"]

        # Rewards
        self.last_health = 1
        self.levels = config["start_level"]
        self.start_level = config["start_level"]
        self.render_reward = config["render_reward"]
        self.exp_reward = 0
        self.nb_badges = 0
        self.pok_count = 1
        self.died = False
        self.rew_norm = 1

        # Exploration memory
        self.sim_frame_dist = config["sim_frame_dist"]
        self.exploration_memory = ExplorationMemory(config["exp_memory_size"], self.im_dim[0]*self.im_dim[1])

        # Save training video
        self.save_video = config["save_video"]
        self.video_path = config["video_path"]
        if self.save_video:
            self.video_writer = media.VideoWriter(self.video_path, (144, 160))
            self.video_writer.__enter__()

        # Actions
        self.action_mapping = {
            0: WindowEvent.PRESS_ARROW_UP,
            1: WindowEvent.PRESS_ARROW_DOWN,
            2: WindowEvent.PRESS_ARROW_LEFT,
            3: WindowEvent.PRESS_ARROW_RIGHT,
            4: WindowEvent.PRESS_BUTTON_A,
            5: WindowEvent.PRESS_BUTTON_B,
        }
        self.release_mapping = {
            0: WindowEvent.RELEASE_ARROW_UP,
            1: WindowEvent.RELEASE_ARROW_DOWN,
            2: WindowEvent.RELEASE_ARROW_LEFT,
            3: WindowEvent.RELEASE_ARROW_RIGHT,
            4: WindowEvent.RELEASE_BUTTON_A,
            5: WindowEvent.RELEASE_BUTTON_B,
        }

        self.data_info = pd.DataFrame(columns=["current_hp","levels", "badges", "pokedex_count", "m", "x", "y"])


    @staticmethod
    def bit_count(bits):
        """
        Count the number of set bits (1s) in a binary number.

        Parameters:
        - bits (int): Binary number.

        Returns:
        - int: Number of set bits.
        """
        return bin(bits).count('1')

    def read_m(self, addr):
        """
        Read a memory value from PyBoy's memory.

        Parameters:
        - addr (int): Memory address.

        Returns:
        - int: Memory value.
        """
        return self.pyboy.get_memory_value(addr)

    def update_exp_memory(self, new_frame):
        vec = new_frame.flatten()
        dist, _ = self.exploration_memory.knn_query(vec, k=1)
        if np.all(self.exploration_memory.frames[0] == 0) :
            self.exploration_memory.update_memory(vec)
            self.exp_reward = 1

        elif dist[0][0][0] > self.sim_frame_dist :
            self.exploration_memory.update_memory(vec)
            self.exp_reward = 1
        
        else :
            self.exp_reward = 0
    
    def get_mxy_coordinates(self):
        """
        Retrieve the coordinates of the player.

        Returns:
        - int: Coordinates of the player
        """
        return (self.bit_count(self.read_m(MAP_N_ADDRESS)), 
            self.bit_count(self.read_m(X_POS_ADDRESS)), 
            self.bit_count(self.read_m(Y_POS_ADDRESS))
        )

    def get_map_location(self, map_idx):
        map_locations = {
            0: "Pallet Town",
            1: "Viridian City",
            2: "Pewter City",
            3: "Cerulean City",
            12: "Route 1",
            13: "Route 2",
            14: "Route 3",
            15: "Route 4",
            33: "Route 22",
            37: "Red house first",
            38: "Red house second",
            39: "Blues house",
            40: "oaks lab",
            41: "Pokémon Center (Viridian City)",
            42: "Poké Mart (Viridian City)",
            43: "School (Viridian City)",
            44: "House 1 (Viridian City)",
            47: "Gate (Viridian City/Pewter City) (Route 2)",
            49: "Gate (Route 2)",
            50: "Gate (Route 2/Viridian Forest) (Route 2)",
            51: "viridian forest",
            52: "Pewter Museum (floor 1)",
            53: "Pewter Museum (floor 2)",
            54: "Pokémon Gym (Pewter City)",
            55: "House with disobedient Nidoran♂ (Pewter City)",
            56: "Poké Mart (Pewter City)",
            57: "House with two Trainers (Pewter City)",
            58: "Pokémon Center (Pewter City)",
            59: "Mt. Moon (Route 3 entrance)",
            60: "Mt. Moon",
            61: "Mt. Moon",
            68: "Pokémon Center (Route 4)",
            193: "Badges check gate (Route 22)"
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return "Unknown Location"

    def get_badges(self):
        """
        Count the number of obtained badges.

        Returns:
        - int: Number of obtained badges.
        """
        return self.bit_count(self.read_m(BADGE_COUNT_ADDRESS))

    def pokedex_count(self):
        """
        Count the number of entries in the Pokedex.

        Returns:
        - int: Number of Pokedex entries.
        """
        return sum([self.bit_count(self.read_m(addr)) for addr in POKEDEX])

    def get_levels(self):
        """
        Get the sum of levels from various addresses.

        Returns:
        - int: Sum of player levels.
        """
        return sum([self.read_m(lev) for lev in LEVELS_ADDRESSES])

    def read_hp_fraction(self):
        """
        Read the player's health fraction.

        Returns:
        - float: Player's health fraction.
        """
        def read_hp(start):
            return 256 * self.read_m(start) + self.read_m(start+1)

        hp_sum = sum([read_hp(add) for add in HP_ADDRESSES])
        max_hp_sum = sum([read_hp(add) for add in MAX_HP_ADDRESSES])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def _get_obs(self):
        """
        Get the current state of the environment.

        Returns:
        - Tuple[Dict, np.ndarray]: Tuple containing a dictionary of environment state and an array representing the screen image.
        """
        screen_image = self.pyboy.screen_image()
        frame = np.array(screen_image.resize(self.resize_shape).convert('L'))

        curr_hp = self.read_hp_fraction()
        if curr_hp == 0 and self.last_health > 0:
            self.died = True
        return frame
    
    def _get_info(self):
        """
        Get the current environment information.

        Returns:
        - Dict: Dictionary containing environment information.
        """
        m, x, y = self.get_mxy_coordinates()
        return {"current_hp" : self.read_hp_fraction(), 
                "levels": self.get_levels(),
                "badges" : self.get_badges(),
                "pokedex_count":self.pokedex_count(),
                "m": m, 
                "x": x, 
                "y": y}
    
    def _get_reward(self):
        """
        Calculate the reward based on the current observation.

        Parameters:
        - obs (Dict): Current environment observation.

        Returns:
        - float: Reward value.
        """
        # Levels
        infos = self._get_info()
        level_reward = max(infos['levels'] - self.levels, 0) 
        self.levels = infos['levels']

        # Badges
        badge_reward = 0
        if infos['badges'] > self.nb_badges :
            badge_reward = 20 
            self.nb_badges = infos['badges']

        # Death
        death_reward = 0
        if self.died :
            death_reward = -3
            self.died = False

        # Pokedex
        pok_reward = 0
        if infos['pokedex_count'] > self.pok_count :
            pok_reward = 5
            self.pok_count = infos['pokedex_count']

        # Levels
        level_reward = max(infos['levels'] - self.levels, 0)
        self.levels = infos['levels']

        if self.render_reward:
            print(f"Pokédex: {pok_reward}, Badges: {badge_reward}, Death: {death_reward}, Levels: {level_reward}, exploration: {self.exp_reward}")
        
        return (pok_reward + badge_reward + death_reward + level_reward + self.exp_reward) / self.rew_norm
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        self.last_health = 1
        self.levels = self.start_level
        self.exploration_memory = ExplorationMemory(20_000, self.im_dim[0]*self.im_dim[1])
        self.done = False
        self.nb_step = 0
        self.exp_reward = 0
        self.nb_badges = 0
        self.pok_count = 1
        if self.save_video:
            self.video_writer = media.VideoWriter(self.video_path, (144, 160))
            self.video_writer.__enter__()
        self.data_info = pd.DataFrame(columns=["current_hp","levels", "badges", "pokedex_count", "m", "x", "y"])
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        if action < 6:
            self.pyboy.send_input(self.action_mapping[action])

        for i in range(self.action_freq-1):
            if i == 8:
                if action < 6:
                    self.pyboy.send_input(self.release_mapping[action])
            self.pyboy.tick()
            # if self.save_video :
            #     self.video_writer.add_image(np.array(self.pyboy.screen_image()))
        observation = self._get_obs()
        info = self._get_info()
        self.data_info.loc[len(self.data_info)] = info
        reward = self._get_reward()
        self.nb_step += 1
        truncated = self.nb_step >= self.max_step
        self.last_health = self.read_hp_fraction()
        terminated = self.last_health == 0 or truncated

        return observation, reward, terminated, truncated, info
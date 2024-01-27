import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy, WindowEvent
from memory_addresses import *

class PokemonBlueEnv(gym.Env):
    def __init__(self, rom_path, emulation_speed=4, start_level=5, render_reward=False):
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
        super(PokemonBlueEnv, self).__init__()

        self.init_state = open("jeu/experimental_states/3pokemon.state", "rb")
        self.pyboy = PyBoy(rom_path)
        self.pyboy.set_emulation_speed(emulation_speed)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8)
        self.last_health = 1
        self.died_count = 0
        self.levels = start_level
        self.start_level = start_level
        self.render_reward = render_reward

        self.level_reward = 0

        self.action_mapping = {
            1: WindowEvent.PRESS_ARROW_UP,
            2: WindowEvent.PRESS_ARROW_DOWN,
            3: WindowEvent.PRESS_ARROW_LEFT,
            4: WindowEvent.PRESS_ARROW_RIGHT,
            5: WindowEvent.PRESS_BUTTON_A,
            6: WindowEvent.PRESS_BUTTON_B,
        }
        self.release_mapping = {
            1: WindowEvent.RELEASE_ARROW_UP,
            2: WindowEvent.RELEASE_ARROW_DOWN,
            3: WindowEvent.RELEASE_ARROW_LEFT,
            4: WindowEvent.RELEASE_ARROW_RIGHT,
            5: WindowEvent.RELEASE_BUTTON_A,
            6: WindowEvent.RELEASE_BUTTON_B,
        }

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

    def get_current_state(self):
        """
        Get the current state of the environment.

        Returns:
        - Tuple[Dict, np.ndarray]: Tuple containing a dictionary of environment state and an array representing the screen image.
        """
        curr_hp = self.read_hp_fraction()
        if curr_hp == 0 and self.last_health > 0:
            self.died_count += 1
        return {
            "hp": curr_hp,
            "levels": self.get_levels(),
            "badges": self.get_badges(),
            "pokedex": self.pokedex_count(),
            "died": self.died_count
        }, np.array(self.pyboy.screen_image)

    def get_reward(self, obs):
        """
        Calculate the reward based on the current observation.

        Parameters:
        - obs (Dict): Current environment observation.

        Returns:
        - float: Reward value.
        """
        self.level_reward = max(obs["levels"] - self.levels, 0)
        if self.render_reward:
            print(f"Pokédex: {5*obs['pokedex']}, Badges: {20*obs['badges']}, Death: {-3*obs['died']}, Levels: {2*max(0, obs['levels']-self.levels)}")
        return 5*obs['pokedex'] + 20*obs["badges"] - 3*obs["died"] + 2*self.level_reward

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - Tuple[Dict, np.ndarray]: Initial environment state.
        """
        self.pyboy.load_state(self.init_state)
        self.last_health = 1
        self.died_count = 0
        self.levels = self.start_level
        self.level_reward = 0
        obs = self.get_current_state()
        return obs

    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Parameters:
        - action (int): Action to take.

        Returns:
        - Tuple[Dict, np.ndarray, float]: Tuple containing the next environment state, screen image, and reward.
        """
        if action > 0:
            self.pyboy.send_input(self.action_mapping[action])

        self.pyboy.tick()

        if action > 0:
            self.pyboy.send_input(self.release_mapping[action])

        obs = self.get_current_state()

        reward = self.get_reward(obs[0])

        self.last_health = self.read_hp_fraction()
        self.levels = obs[0]["levels"]

        return obs[0], obs[1], reward

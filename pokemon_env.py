import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy, WindowEvent
from memory_addresses import *
from memory import ExplorationMemory
import yaml

class PokemonEnv(gym.Env):
    def __init__(self, config_file="env_config.yaml"):
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
        super().__init__()

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.init_state = config["init_state"]
        if config["render_view"] : 
            self.pyboy = PyBoy(config["rom_path"])
        else :
            self.pyboy = PyBoy(config["rom_path"], window_type="headless")
        self.pyboy.set_emulation_speed(config["emulation_speed"])
        self.observation_space = spaces.Box(low=0, high=255, shape=config["im_dim"],  dtype=np.uint8)
        self.done = False
        self.resize_shape = (config["im_dim"][1], config["im_dim"][0])
        self.nb_step = 0
        self.max_step = config['ep_length']
        self.action_freq = 48
        self.im_dim = config["im_dim"]

        # Rewards
        self.last_health = 1
        self.died_count = 0
        self.levels = config["start_level"]
        self.start_level = config["start_level"]
        self.render_reward = config["render_reward"]
        self.level_reward = 0
        self.exp_reward = 0

        # Exploration memory
        self.sim_frame_dist = config["sim_frame_dist"]
        self.exploration_memory = ExplorationMemory(config["exp_memory_size"], self.im_dim[0]*self.im_dim[1])

        # Actions
        self.action_space = spaces.Discrete(config["nb_action"]) # 6: no action
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
        if self.exp_reward == 0 :
            self.exploration_memory.update_memory(vec)
            self.exp_reward += 1

        elif dist > self.sim_frame_dist :
            self.exploration_memory.update_memory(vec)
            self.exp_reward += 1


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
        frame = np.array(self.pyboy.screen_image().resize(self.resize_shape).convert('L'))
        curr_hp = self.read_hp_fraction()
        if curr_hp == 0 and self.last_health > 0:
            self.died_count += 1
        return [curr_hp, self.get_levels(), self.get_badges(), self.pokedex_count()], frame


    def get_reward(self, infos):
        """
        Calculate the reward based on the current observation.

        Parameters:
        - obs (Dict): Current environment observation.

        Returns:
        - float: Reward value.
        """
        self.level_reward = max(infos[1] - self.levels, 0)
        if self.render_reward:
            print(f"Pokédex: {5*infos[3]}, Badges: {20*infos[2]}, Death: {-3*self.died_count}, Levels: {2*self.level_reward}, exploration: {5*self.exp_reward}")
        return 5*infos[3] + 20*infos[2] - 3*self.died_count + 2*self.level_reward + 5*self.exp_reward


    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - Tuple[Dict, np.ndarray]: Initial environment state.
        """
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        self.last_health = 1
        self.died_count = 0
        self.levels = self.start_level
        self.level_reward = 0
        self.exploration_memory = ExplorationMemory(20_000, self.im_dim[0]*self.im_dim[1])
        obs = self.get_current_state()
        self.done = False
        self.nb_step = 0
        return obs[0], obs[1]


    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Parameters:
        - action (int): Action to take.

        Returns:
        - Tuple[Dict, np.ndarray, float]: Tuple containing the next environment state, screen image, and reward.
        """
        if action < 6:
            self.pyboy.send_input(self.action_mapping[action])

        self.pyboy.tick()

        if action < 6:
            self.pyboy.send_input(self.release_mapping[action])

        for _ in range(self.action_freq-1):
            self.pyboy.tick()

        obs = self.get_current_state()
        self.update_exp_memory(obs[1])

        reward = self.get_reward(obs[0])

        self.last_health = self.read_hp_fraction()
        self.levels = obs[0][1]

        self.nb_step += 1
        if self.nb_step >= self.max_step:
            self.done = True

        return obs[0], obs[1], reward, self.done
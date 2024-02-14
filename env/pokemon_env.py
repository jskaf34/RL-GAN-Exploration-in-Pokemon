import gym
from gym import spaces
import numpy as np
from pyboy import PyBoy, WindowEvent
from .memory_addresses import *
from .memory import ExplorationMemory
import yaml
import mediapy as media

class PokemonEnv(gym.Env):
    def __init__(self, config_file="env_config.yaml"):
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
        self.action_freq = config['action_freq']
        self.im_dim = config["im_dim"]

        # Rewards
        self.last_health = 1
        self.experience_sum = config["start_experience"]
        self.start_experience = config["start_experience"]
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
    
    def get_coordinates(self):
        coords = (self.read_m(X_POS_ADDRESS), self.read_m(Y_POS_ADDRESS))
        map_n = self.read_m(MAP_N_ADDRESS)
        return map_n, coords

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

    def get_badges(self):
        """
        Count the number of obtained badges.

        Returns:
        - int: Number of obtained badges.
        """
        return self.bit_count(self.read_m(BADGE_COUNT_ADDRESS))

    def pokedex_count(self):
        return sum([self.bit_count(self.read_m(addr)) for addr in POKEDEX])
    
    def get_exp(self):
        return sum([self.read_m(exp) for exp in POKEMON_EXP])

    def read_hp_fraction(self):
        def read_hp(start):
            return 256 * self.read_m(start) + self.read_m(start+1)

        hp_sum = sum([read_hp(add) for add in HP_ADDRESSES])
        max_hp_sum = sum([read_hp(add) for add in MAX_HP_ADDRESSES])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def get_current_state(self):
        screen_image = self.pyboy.screen_image()
        frame = np.array(screen_image.resize(self.resize_shape).convert('L'))
        curr_hp = self.read_hp_fraction()
        if curr_hp == 0 and self.last_health > 0:
            self.died = True
        return [curr_hp, self.get_exp(), self.get_badges(), self.pokedex_count()], frame

    def get_reward(self, infos):
        # Pokemon experience
        experience_ratio = (infos[1] - self.experience_sum) / self.experience_sum
        experience_reward = max(experience_ratio, 0) * 5
        self.experience_sum = infos[1]

        # Badges
        badge_reward = 0
        if infos[2] > self.nb_badges :
            badge_reward = 30 
            self.nb_badges = infos[2]

        # Death
        death_reward = 0
        if self.died :
            death_reward = -3
            self.died = False

        # Pokedex
        pok_reward = 0
        if infos[3] > self.pok_count :
            pok_reward = 10
            self.pok_count = infos[3]

        if self.render_reward:
            print(f"Pokédex: {pok_reward}, Badges: {badge_reward}, Death: {death_reward}, Experiences: {experience_reward}, exploration: {self.exp_reward}")
        
        return (pok_reward + badge_reward + death_reward + experience_reward + self.exp_reward) / self.rew_norm

    def reset(self):
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        self.last_health = 1
        self.experience_sum = self.start_experience
        self.exploration_memory = ExplorationMemory(20_000, self.im_dim[0]*self.im_dim[1])
        obs = self.get_current_state()
        self.done = False
        self.nb_step = 0
        self.exp_reward = 0
        self.nb_badges = 0
        self.pok_count = 1
        if self.save_video:
            self.video_writer = media.VideoWriter(self.video_path, (144, 160))
            self.video_writer.__enter__()
        return obs[0], obs[1]

    def step(self, action):
        if action < 6:
            self.pyboy.send_input(self.action_mapping[action])

        for i in range(self.action_freq-1):
            if i == 8:
                if action < 6:
                    self.pyboy.send_input(self.release_mapping[action])
            self.pyboy.tick()
            if self.save_video :
                self.video_writer.add_image(np.array(self.pyboy.screen_image()))

        obs = self.get_current_state()
        self.update_exp_memory(obs[1])

        reward = self.get_reward(obs[0])

        self.last_health = self.read_hp_fraction()

        self.nb_step += 1
        if self.nb_step >= self.max_step:
            self.done = True

        return obs[0], obs[1], reward, self.done
import numpy as np
from scipy.spatial.distance import cdist
from collections import namedtuple, deque
import random

class FrameStacker:
    def __init__(self, num_frames, frame_shape):
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.frames_buffer = deque(maxlen=num_frames)

    def reset(self, initial_frame):
        for _ in range(self.num_frames):
            self.frames_buffer.append(initial_frame)

    def stack_frames(self):
        return np.stack(self.frames_buffer, axis=-1)

    def update_buffer(self, new_frame):
        self.frames_buffer.append(new_frame)


class ExplorationMemory:
    def __init__(self, capacity, vec_frame_shape):
        self.capacity = capacity
        self.frame_shape = vec_frame_shape
        self.frames = np.zeros((capacity,) + (vec_frame_shape,), dtype=np.uint8)
        self.current_index = 0
        self.is_full = False

    def update_memory(self, new_frame):
        self.frames[self.current_index] = new_frame
        self.current_index = (self.current_index + 1) % self.capacity
        if not self.is_full and self.current_index == 0:
            self.is_full = True

    def get_frames(self):
        if self.is_full:
            return self.frames
        else:
            return self.frames[: self.current_index + 1]

    def knn_query(self, query_frame, k=1):
        frames_to_query = self.get_frames()
        flattened_frames = frames_to_query.reshape((frames_to_query.shape[0], -1))

        distances = cdist(flattened_frames, [query_frame], metric='euclidean')
        indices = np.argsort(distances, axis=0)[:k]

        return distances[indices], indices


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('stacked_frames', 'state_infos', 'action', 'next_stacked_frames', 'next_state_infos', 'reward', 'done'))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
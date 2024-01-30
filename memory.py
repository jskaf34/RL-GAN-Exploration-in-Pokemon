import numpy as np
from scipy.spatial.distance import cdist

class FrameMemory:
    def __init__(self, num_frames, frame_shape):
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.memory = np.zeros((num_frames, *frame_shape), dtype=np.uint8)
        self.current_index = 0

    def update_memory(self, new_frame):
        self.memory[self.current_index] = new_frame
        self.current_index = (self.current_index + 1) % self.num_frames

    def get_recent_frames(self):
        return self.memory

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
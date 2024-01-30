from memory import ExplorationMemory
import numpy as np
import time
import hnswlib
from tqdm import tqdm

def fill_custom_memory(memory, num_samples=20_000):
    for _ in tqdm(range(num_samples)):
        frame = np.random.rand(144, 160, 3).flatten()
        memory.update_memory(frame)

def fill_opt_memory(opt_memory, num_samples=20_000):
    for _ in tqdm(range(num_samples)):
        frame = np.random.rand(144, 160, 3).flatten()
        opt_memory.add_items(frame, np.array([opt_memory.get_current_count()]))

def search_custom_memory(memory, query_frame):
    start_time = time.time()
    dist, ind = memory.knn_query(query_frame)
    duration = time.time() - start_time
    print(f"Custom search time: {duration}")
    return dist, ind

def search_opt_memory(opt_memory, query_frame):
    start_time = time.time()
    index, dist_opt = opt_memory.knn_query(query_frame, k=1)
    duration = time.time() - start_time
    print(f"Opt search time: {duration}")
    return index, dist_opt

def main():
    custom_memory = ExplorationMemory(20_000, 144*160*3)
    opt_memory = hnswlib.Index(space='l2', dim=144*160*3)
    opt_memory.init_index(max_elements=20_000, ef_construction=100, M=16)

    # Fill memory
    fill_custom_memory(custom_memory)
    fill_opt_memory(opt_memory)

    # Search time
    query_frame = np.random.rand(144, 160, 3).flatten()

    custom_dist, custom_ind = search_custom_memory(custom_memory, query_frame)
    opt_index, opt_dist = search_opt_memory(opt_memory, query_frame)

    print(custom_dist, opt_dist, custom_ind, opt_index)

if __name__ == "__main__":
    main()
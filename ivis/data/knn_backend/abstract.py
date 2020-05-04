import time
from collections import namedtuple
from multiprocessing import cpu_count, Queue
from operator import attrgetter

import numpy as np
from tqdm import tqdm

IndexNeighbours = namedtuple('IndexNeighbours', 'row_index neighbour_list')


class IndexBuildingError(OSError):
    pass


class KnnBackend:
    def __init__(self, index, shape, verbose=1):
        self.index = index
        self.shape = shape
        self.verbose = verbose

    def extract_knn(self, worker_cls, k=150, search_k=0):
        """ Starts multiple processes to retrieve nearest neighbours using
            either an Annoy Index or NGT index, in parallel. """
        n_dims = self.shape[1]
        threads = cpu_count() - 1
        chunk_size = self.shape[0] // threads
        remainder = (self.shape[0] % threads) > 0
        process_pool = []
        results_queue = Queue()

        # Split up the indices and assign processes for each chunk
        i = 0
        while (i + chunk_size) <= self.shape[0]:
            process_pool.append(worker_cls(self.index, k, search_k, n_dims,
                                           (i, i+chunk_size), results_queue))
            i += chunk_size
        if remainder:
            batch_shape = (i, self.shape[0])
            process_pool.append(worker_cls(self.index, k, search_k, n_dims,
                                           batch_shape, results_queue))

        try:
            for process in process_pool:
                process.start()

            # Read from queue constantly to prevent it from becoming full
            with tqdm(total=self.shape[0], disable=self.verbose < 1) as pbar:
                neighbours = []
                neighbour_lst_len = len(neighbours)
                while any(process.is_alive() for process in process_pool):
                    while not results_queue.empty():
                        neighbours.append(results_queue.get())
                    progress = len(neighbours) - neighbour_lst_len
                    pbar.update(progress)
                    neighbour_lst_len = len(neighbours)
                    time.sleep(2)

                while not results_queue.empty():
                    neighbours.append(results_queue.get())

            neighbours = sorted(neighbours, key=attrgetter('row_index'))
            neighbours = list(map(attrgetter('neighbour_list'), neighbours))

            return np.array(neighbours)

        except Exception:
            print('Halting KNN retrieval and cleaning up')
            for process in process_pool:
                process.terminate()
            raise

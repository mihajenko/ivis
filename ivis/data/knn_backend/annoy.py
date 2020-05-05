""" KNN retrieval using an Annoy index. """
import os
from multiprocessing import Process

import numpy as np
from annoy import AnnoyIndex
from scipy.sparse import issparse
from tqdm import tqdm

from ivis.data.knn_backend.abstract import IndexBuildingError, IndexNeighbours
from ivis.data.knn_backend.abstract import KnnBackend


class AnnoyBackend(KnnBackend):
    def __init__(self, X, index_filepath, distance_metric='angular', ntrees=50,
                 build_index_on_disk=False, verbose=1):
        """
        :param array X: numpy array with shape (n_samples, n_features)
        :param str index_filepath: The filepath of a trained annoy index file
            saved on disk.
        :param distance_metric: The distance metric supported by Annoy.
        :param int ntrees: The number of random projections trees built by
            Annoy to approximate KNN. The more trees the higher the memory
            usage, but the better the accuracy of results.
        :param bool build_index_on_disk: Whether to build the annoy index
            directly on disk. Building on disk should allow for bigger
            datasets to be indexed, but may cause issues. If None, on-disk
            building will be enabled for Linux, but not Windows due to
            issues on Windows.
        :param int verbose: Controls the volume of logging output the model
            produces when training. When set to 0, silences outputs, when
            above 0 will print outputs.
        """
        self.X = X
        self.build_index_on_disk = build_index_on_disk
        self.distance_metric = distance_metric
        self.index_filepath = index_filepath
        self.ntrees = ntrees

        self.index = AnnoyIndex(self.X.shape[1], metric=self.distance_metric)

        if self.index_filepath and os.path.exists(self.index_filepath):
            self.index.load(self.index_filepath)

        else:  # build a standalone annoy index
            if verbose > 0:
                print('Building KNN index')

            if self.build_index_on_disk:
                self.index.on_disk_build(self.index_filepath)

            if issparse(self.X):
                for i in tqdm(range(self.X.shape[0]), disable=verbose < 1):
                    v = self.X[i].toarray()[0]
                    self.index.add_item(i, v)
            else:
                for i in tqdm(range(self.X.shape[0]), disable=verbose < 1):
                    v = self.X[i]
                    self.index.add_item(i, v)

        try:
            self.index.build(self.ntrees)
        except Exception:
            msg = ("Error building Annoy Index. Passing on_disk_build=False"
                   " may solve the issue, especially on Windows.")
            raise IndexBuildingError(msg)

        super().__init__(self.index, self.X.shape, verbose=verbose)


class AnnoyKnnWorker(Process):
    """
    Upon construction, this worker process loads an annoy index from disk.
    When started, the neighbours of the data-points specified by `data_indices`
    will be retrieved from the index according to the provided parameters
    and stored in the 'results_queue'.

    `data_indices` is a tuple of integers denoting the start and end range of
    indices to retrieve.
    """
    def __init__(self, index_filepath, k, search_k, n_dims,
                 data_indices, results_queue):
        self.index_filepath = index_filepath
        self.k = k
        self.n_dims = n_dims
        self.search_k = search_k
        self.data_indices = data_indices
        self.results_queue = results_queue
        super().__init__()

    def run(self):
        try:
            index = AnnoyIndex(self.n_dims, metric='angular')
            index.load(self.index_filepath)
            for i in range(self.data_indices[0], self.data_indices[1]):
                neighbour_indexes = index.get_nns_by_item(
                    i, self.k, search_k=self.search_k, include_distances=False)
                neighbour_indexes = np.array(neighbour_indexes,
                                             dtype=np.uint32)
                self.results_queue.put(
                    IndexNeighbours(row_index=i,
                                    neighbour_list=neighbour_indexes))
        except Exception as e:
            self.exception = e

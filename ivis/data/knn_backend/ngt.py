""" KNN retrieval using an NGT index. """

import ngtpy
import numpy as np
from multiprocessing import cpu_count
from operator import attrgetter
from scipy.sparse import issparse
from tqdm import tqdm

from .abstract import KnnBackend
from .abstract import IndexNeighbours, IndexBuildingError


class NGTBackend(KnnBackend):
    def __init__(self, X, index_filepath, distance_metric='Jaccard', ntrees=50,
                 verbose=1):
        """ Build a standalone NGT index. UNIX systems, on-disk only.

        :param array X: numpy array with shape (n_samples, n_features)
        :param str index_filepath: The filepath of a trained NGT index file
            saved on disk.
        :param distance_metric: The distance metric supported by NGT. Choose
            from: "L1", "L2", "Hamming", "Jaccard", "Angle",
            "Normalized Angle", "Cosine", "Normalized Cosine"
        :param int ntrees: The number of random projections trees built by NGT
            to approximate KNN. The more trees the higher the memory usage,
            but the better the accuracy of results.
        :param int verbose: Controls the volume of logging output the model
            produces when training. When set to 0, silences outputs, when
            above 0 will print outputs.
        """
        super().__init__()
        self.X = X
        self.path = index_filepath
        self.distance_metric = distance_metric
        self.ntrees = ntrees
        self.verbose = verbose

    def load_index(self):
        return ngtpy.Index(self.path)

    def build_index(self):
        ngtpy.create(self.path, self.X.shape[1],
                     edge_size_for_creation=self.ntrees,
                     distance_type=self.distance_metric)
        index = ngtpy.Index(self.path)

        if issparse(self.X):
            for i in tqdm(range(self.X.shape[0]), disable=self.verbose < 1):
                # periodic save
                if (i % 1000) == 0:
                    index.save()

                v = self.X[i].toarray()[0]
                index.insert(v)
        else:
            for i in tqdm(range(self.X.shape[0]), disable=self.verbose < 1):
                # batch save
                if (i % 1000) == 0:
                    index.save()

                v = self.X[i]
                index.insert(v)

        # final save
        index.save()

        try:
            index.build_index(num_threads=cpu_count() - 1)
        except Exception:
            msg = "Error building NGT Index."
            raise IndexBuildingError(msg)

        return index

    def extract_knn(self, k=150, search_k=0):
        """ Starts multiple processes to retrieve nearest neighbours using
            an NGT Index in parallel """

        index = ngtpy.Index(self.path, read_only=True)

        neighbour_lst = []
        for i in tqdm(range(self.X.shape[0]), disable=self.verbose < 1):
            row = self.X[i, :]
            neighbour_indexes = index.search(row, size=k, edge_size=search_k,
                                             with_distance=False)
            neighbour_indexes = np.array(neighbour_indexes, dtype=np.uint32)
            neighbour_lst.append(
                IndexNeighbours(row_index=i, neighbour_list=neighbour_indexes))

        neighbour_lst = sorted(neighbour_lst, key=attrgetter('row_index'))
        neighbour_lst = list(map(attrgetter('neighbour_list'), neighbour_lst))

        return np.array(neighbour_lst)

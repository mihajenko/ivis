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
        if self.distance_metric in ('Hamming', 'Jaccard'):
            object_type = 'Byte'
        else:
            object_type = 'Float'
        ngtpy.create(self.path, self.X.shape[1],
                     edge_size_for_creation=self.ntrees,
                     distance_type=self.distance_metric,
                     object_type=object_type)
        index = ngtpy.Index(self.path)

        if issparse(self.X):
            for i in tqdm(range(self.X.shape[0]), disable=self.verbose < 1):
                # periodically save the inserts
                if (i % 1000) == 0:
                    index.save()

                v = self.X[i].toarray()[0]
                index.insert(v)
        else:
            for i in tqdm(range(self.X.shape[0]), disable=self.verbose < 1):
                # periodically save the inserts
                if (i % 1000) == 0:
                    index.save()

                v = self.X[i]
                index.insert(v)

        # save the final inserts
        index.save()

        try:
            index.build_index(num_threads=cpu_count() - 1)
            # save the KNN index
            index.save()
        except Exception:
            msg = "Error building NGT Index."
            raise IndexBuildingError(msg)
        finally:
            index.close()

    def extract_knn(self, k=150, search_k=0):
        """ Starts multiple processes to retrieve nearest neighbours using
            an NGT Index in parallel """

        index = ngtpy.Index(self.path, read_only=True)

        neighbours = []
        try:
            for i in tqdm(range(self.X.shape[0]), disable=self.verbose < 1):
                row = self.X[i, :]
                neighbour_indices = index.search(row,
                                                 size=k,
                                                 edge_size=search_k,
                                                 with_distance=False)
                neighbour_indices = np.array(neighbour_indices,
                                             dtype=np.uint32)
                neighbours.append(
                    IndexNeighbours(row_index=i,
                                    neighbour_list=neighbour_indices))
        except Exception:
            msg = "Error extracting KNN tree."
            raise IndexBuildingError(msg)
        else:
            neighbours = sorted(neighbours, key=attrgetter('row_index'))
            neighbours = list(map(attrgetter('neighbour_list'), neighbours))
            return np.array(neighbours)
        finally:
            index.close()

""" KNN retrieval using an NGT index. """
from multiprocessing import Process, cpu_count

import ngtpy
import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm

from .abstract import IndexNeighbours, IndexBuildingError
from .abstract import KnnBackend


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
        super().__init__(X, index_filepath, verbose=verbose)
        self.distance_metric = distance_metric
        self.ntrees = ntrees

    def load_index(self):
        return ngtpy.Index(self.index_filepath)

    def build_index(self):
        if self.distance_metric in ('Hamming', 'Jaccard'):
            object_type = 'Byte'
        else:
            object_type = 'Float'
        ngtpy.create(self.index_filepath, self.X.shape[1],
                     edge_size_for_creation=self.ntrees,
                     distance_type=self.distance_metric,
                     object_type=object_type)
        index = ngtpy.Index(self.index_filepath)

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


class NGTKnnWorker(Process):
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
        super(NGTKnnWorker, self).__init__()

    def run(self):
        index = ngtpy.Index(self.index_filepath, read_only=True)
        try:
            for i in range(self.data_indices[0], self.data_indices[1]):
                row = index.get_object(i)
                neighbour_indices = index.search(row,
                                                 size=self.k,
                                                 edge_size=self.search_k,
                                                 with_distance=False)
                neighbour_indices = np.array(neighbour_indices,
                                             dtype=np.uint32)
                self.results_queue.put(
                    IndexNeighbours(row_index=i,
                                    neighbour_list=neighbour_indices))
        except Exception as e:
            self.exception = e
        finally:
            self.results_queue.close()
            index.close()

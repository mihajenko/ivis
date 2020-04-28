from abc import ABC, abstractmethod
from collections import namedtuple


IndexNeighbours = namedtuple('IndexNeighbours', 'row_index neighbour_list')


class IndexBuildingError(OSError):
    pass


class KnnBackend(ABC):
    def __init__(*args, **kwargs):
        pass

    @abstractmethod
    def load_index(self, *args, **kwargs):
        pass

    @abstractmethod
    def build_index(self, *args, **kwargs):
        pass

    @abstractmethod
    def extract_knn(self, *args, **kwargs):
        pass

import ngtpy
import numpy as np
import os
import pytest
import shutil
import tempfile
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix
from sklearn import datasets

from ivis.data.knn_backend.annoy import AnnoyBackend
from ivis.data.knn_backend.ngt import NGTBackend


@pytest.fixture(scope='function')
def annoy_index_file():
    _, filepath = tempfile.mkstemp('.index')
    yield filepath
    os.remove(filepath)


@pytest.fixture(scope='function')
def ngt_index_file():
    filepath = tempfile.mkdtemp()
    yield filepath
    shutil.rmtree(filepath)


def test_build_sparse_annoy_index(annoy_index_file):
    data = np.random.choice([0, 1], size=(10, 5))
    sparse_data = csr_matrix(data)

    annoy_backend = AnnoyBackend(sparse_data, annoy_index_file)
    index = annoy_backend.build_index(build_index_on_disk=False)

    assert os.path.exists(annoy_index_file)

    loaded_index = AnnoyIndex(5, metric='angular')
    loaded_index.load(annoy_index_file)

    assert index.f == loaded_index.f == 5
    assert index.get_n_items() == loaded_index.get_n_items() == 10
    assert index.get_nns_by_item(0, 5) == loaded_index.get_nns_by_item(0, 5)

    index.unload()
    loaded_index.unload()


def test_dense_annoy_index(annoy_index_file):
    data = np.random.choice([0, 1], size=(10, 5))

    annoy_backend = AnnoyBackend(data, annoy_index_file)
    index = annoy_backend.build_index(build_index_on_disk=False)

    assert os.path.exists(annoy_index_file)

    loaded_index = AnnoyIndex(5, metric='angular')
    loaded_index.load(annoy_index_file)

    assert index.f == loaded_index.f == 5
    assert index.get_n_items() == loaded_index.get_n_items() == 10
    assert index.get_nns_by_item(0, 5) == loaded_index.get_nns_by_item(0, 5)

    index.unload()
    loaded_index.unload()


def test_index_with_ngt(ngt_index_file):
    data = np.random.choice([0, 1], size=(10, 5))
    ngt_backend = NGTBackend(data, ngt_index_file, distance_metric='Jaccard',
                             ntrees=5)
    ngt_backend.build_index()
    assert os.path.exists(ngt_index_file)

    loaded_index = ngtpy.Index(ngt_index_file)

    for i in range(data.shape[0]):
        row_i = data[i, :]
        retrieved = np.array(loaded_index.get_object(i), dtype=np.uint8)
        assert np.all(retrieved == row_i)

    loaded_index.close()


def test_knn_retrieval():
    annoy_index_filepath = 'tests/data/.test-annoy-index.index'
    expected_neighbour_list = np.load('tests/data/test_knn_k4.npy')

    iris = datasets.load_iris()
    X = iris.data

    annoy_backend = AnnoyBackend(X, annoy_index_filepath,
                                 distance_metric='angular')
    annoy_backend.load_index()
    neighbour_list = annoy_backend.extract_knn(k=4, search_k=-1)

    assert np.all(expected_neighbour_list == neighbour_list)


def test_knn_retrieval_with_ngt():
    ngt_index_filepath = 'tests/data/.test-ngt-index.index'
    expected_neighbour_list = np.load('tests/data/test_knn_k4_ngt.npy')

    iris = datasets.load_iris()
    X = iris.data

    ngt_backend = NGTBackend(X, ngt_index_filepath, distance_metric='Jaccard',
                             ntrees=50)
    ngt_backend.load_index()
    neighbour_list = ngt_backend.extract_knn(k=4, search_k=-1)

    assert np.all(expected_neighbour_list == neighbour_list)

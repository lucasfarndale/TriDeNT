import cuml
import matplotlib.pyplot as plt
import numpy as np

def compute_umap_embeddings(outputs, n_components=2, n_neighbours=15, min_dist=0., verbose=True, **kwargs):
    reducer = cuml.UMAP(n_components=n_components, n_neighbors=n_neighbours, min_dist=min_dist, verbose=True, **kwargs)
    embedding = reducer.fit_transform(outputs)
    return embedding

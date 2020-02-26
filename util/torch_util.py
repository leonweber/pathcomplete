import numpy as np
import itertools

def multilabels_to_onehot(multilabels, n_labels):
    max_index = max(itertools.chain(*multilabels))
    arr = np.zeros((len(multilabels), n_labels))

    for i, indices in enumerate(multilabels):
        for index in indices:
            arr[i, index] = 1

    return arr



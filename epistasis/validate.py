import numpy as np
import pandas as pd
from .stats import split_gpm, pearson


def k_fold(gpm, model, k=10):
    """Cross-validation using K-fold validation on a seer.
    """
    # Get index.
    idx = np.copy(gpm.index)

    # Shuffle index
    np.random.shuffle(idx)

    # Get subsets.
    subsets = np.array_split(idx, k)
    subsets_idx = np.arange(len(subsets))

    # Do k-fold
    scores = []
    for i in range(k):
        # Split index into train/test subsets
        train_idx = np.concatenate(np.delete(subsets, i))
        test_idx = subsets[i]

        # Split genotype-phenotype map
        train, test = split_gpm(gpm, idx=train_idx)

        # Fit model.
        model.add_gpm(train)
        model.fit()

        # Score validation set
        pobs = test.phenotypes
        pred = model.predict(X=test.genotypes)

        score = pearson(pobs, pred)**2
        scores.append(score)

    return scores


def holdout(gpm, model, size=1, repeat=1):
    """Validate a model by holding-out parts of the data.
    """
    train_scores = []
    test_scores = []

    for i in range(repeat):
        # Get index.
        idx = np.copy(gpm.index)

        # Shuffle index
        np.random.shuffle(idx)

        train_idx = idx[:size]

        # Split genotype-phenotype map
        train, test = split_gpm(gpm, idx=train_idx)

        # Fit model.
        model.add_gpm(train)
        model.fit()
        train_scores.append(model.score())

        # Score validation set
        pobs = test.phenotypes
        pred = model.predict(X=test.genotypes)

        score = pearson(pobs, pred)**2
        test_scores.append(score)

    return train_scores, test_scores

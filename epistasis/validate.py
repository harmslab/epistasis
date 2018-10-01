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

    model.add_gpm(gpm)
    X = model._X()

    for i in range(repeat):
        # Get index.
        idx = np.copy(gpm.index)

        # Shuffle index
        np.random.shuffle(idx)

        # Split model matriay to cross validate).
        train_idx = idx[:size]
        test_idx = idx[size:]
        train_X = X[train_idx, :]
        test_X = X[test_idx, :]

        # Train the model
        model.fit(X=train_X, y=gpm.phenotypes[train_idx])

        train_p = model.predict(X=train_X)
        train_s = pearson(train_p, gpm.phenotypes[train_idx])**2
        train_scores.append(train_s)

        # Test the model
        test_p = model.predict(X=test_X)
        test_s = pearson(test_p, gpm.phenotypes[test_idx])**2
        test_scores.append(test_s)


    return train_scores, test_scores

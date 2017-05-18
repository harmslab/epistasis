import os
import shutil
import h5py
from datetime import datetime
import pickle
import numpy as np

class SamplerError(Exception):
    """Raise an exception from the Sampler class"""

def add_datetime_to_filename(filepath):
    """Inserts a datetime at the end of a filename, before the extension."""
    name, ext = os.path.splitext(filepath)
    # Get current date/time in isoformat
    t = datetime.strftime(datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    return "{:}-{:}{:}".format(*[name, t, ext])

class Sampler(object):
    """A base class to be inherited by sampling classes. Constructs a database for
    storing the samples.
    """
    def __init__(self, model, db_dir=None):
        # Check the model
        self._model = model
        if hasattr(self.model, "gpm") is False:
            raise SamplerError("The Epistasis model must have a GenotypePhenotypeMap as the `gpm` attribute.")

        # -----------------------------------
        # Set up the sampling database
        # -----------------------------------
        if db_dir is None:
            self._db_dir = add_datetime_to_filename("sampler")
        else:
            self._db_dir = db_dir

        # Create a folder for the database.
        if not os.path.exists(self._db_dir):
            os.makedirs(self._db_dir)

        self._db_path = os.path.join(self._db_dir, "sample-db.hdf5")
        self._model_path = os.path.join(self._db_dir, "model.pickle")

        # Create the hdf5 file for saving samples.
        self.File = h5py.File(self._db_path, "a")

        # Write model to db_dir
        with open(self._model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Add database
        if "coefs" not in self.File:
            self.File.create_dataset("coefs", (0,0), maxshape=(None,None), compression="gzip")
        if "scores" not in self.File:
            self.File.create_dataset("scores", (0,), maxshape=(None,), compression="gzip")

    @classmethod
    def from_db(cls, db_dir, overwrite=True):
        """Start from a previously greated Sampler database. This method assumes strict
        structure of the database. First, the sampler file must be an hdf5 file named
        'sample-db.hdf5' and a pickle file with an epistasis model names `model.pickle`.

        Note: currently, won't check to see if the sampling database if bayesian vs. bootstrap.
        """
        with open(os.path.join(db_dir,"model.pickle"), "rb") as f:
            model = pickle.load(f)

        if overwrite:
            self = cls(model, db_dir=db_dir)
        else:
            # New database directory
            new_db_dir = add_datetime_to_filename("sampler")

            # Create a folder for the database.
            if not os.path.exists(new_db_dir):
                os.makedirs(new_db_dir)

            # Old database path
            old_db_path = os.path.join(db_dir, "sample-db.hdf5")

            # Copy the old database to new database
            shutil.copyfile(db_dir, new_db_dir)

            self = cls(model, db_dir=new_db_dir)
        return self

    @property
    def model(self):
        """Get model. Protects the model from being changed once passed to sampler."""
        return self._model

    def write_dataset(self, key, data):
        """Write data to database file.
        """
        # Get the dataset
        ds = self.File[key]

        # Resize the dataset for the new samples
        old_dims = ds.shape
        new_dims = list(data.shape)
        new_dims[0] = old_dims[0] + data.shape[0]
        ds.resize(tuple(new_dims))

        # Add the new samples
        if len(new_dims) == 1:
            ds[old_dims[0]:new_dims[0]] = data
        elif len(new_dims) == 2:
            ds[old_dims[0]:new_dims[0], :] = data

    @property
    def coefs(self):
        """Samples of epistatic coefficients. Rows are samples, Columns are coefs."""
        return self.File["coefs"]

    @property
    def labels(self):
        """Labels of the coefs."""
        return self.model.epistasis.labels

    @property
    def scores(self):
        """Samples of epistatic coefficients. Rows are samples, Columns are coefs."""
        return self.File["scores"]

    @property
    def best_coefs(self):
        """Most probable model."""
        index = np.argmax(self.scores.value)
        return self.coefs[index,:]

    def percentiles(self, percentiles):
        """Return credibility regions (Bayes) or confidence intervals (Bootstrap)."""
        return np.percentile(self.coefs.value, percentiles, axis=0)

    def predict(self, samples):
        """"""
        X = self.model.X_constructor(genotypes=self.model.gpm.complete_genotypes)
        predictions = np.empty((samples.shape[0], len(self.model.gpm.complete_genotypes)), dtype=float)
        for i in range(len(samples)):
            predictions[i,:] = self.model.hypothesis(X=X, thetas=self.coefs[i,:])
        return predictions

    def predict_from_random_samples(self, n):
        """Randomly draw from sampled models and predict phenotypes."""
        sample_size, coef_size = self.coefs.shape
        model_indices = np.random.choice(np.arange(sample_size), n, replace=False)
        samples = np.empty((n, coef_size))
        for i, index in enumerate(model_indices):
            samples[i,:] = self.coefs[index, :]
        return self.predict(samples=samples)

    def predict_from_top_samples(self, n):
        """Draw from top sampled models and predict phenotypes."""
        sample_size, coef_size = self.coefs.shape
        model_indices = np.argsort(self.scores)[::-1]
        samples = np.empty((n, coef_size))
        for i, index in enumerate(model_indices[:n]):
            samples[i,:] = self.coefs[index, :]
        return self.predict(samples=samples)

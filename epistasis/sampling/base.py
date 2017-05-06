import os
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
        self.File = h5py.File(self._db_path, "w")

        # Write model to db_dir
        with open(self._model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Add database
        self.File.create_dataset("coefs", (0,0), maxshape=(None,None), compression="gzip")
        self.File.create_dataset("scores", (0,), maxshape=(None,), compression="gzip")

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
        return np.percentile(self.coefs.value, percentiles, axis=0)

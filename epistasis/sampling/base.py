import os
import h5py
from datetime import datetime
import pickle

class SamplerError(Exception):
    """Raise an exception from the Sampler class"""

def add_datetime_to_filename(filepath):
    """Inserts a datetime at the end of a filename, before the extension."""
    name, ext = os.path.splitext(filepath)
    # Get current date/time in isoformat
    t = datetime.strftime(datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    return "{:}-{:}{:}".format(*[name, t, ext])

class Sampler(object):
    """Base Sampler class.

    Creates a directory that contains information for the model.
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
            self.db_dir = add_datetime_to_filename("sampler")
        else:
            self.db_dir = db_dir

        # Create a folder for the database.
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        self.db_path = os.path.join(self.db_dir, "sample-db.hdf5")
        self.model_path = os.path.join(self.db_dir, "model.pickle")

        # Create the hdf5 file for saving samples.
        self.File = h5py.File("sample-db.hdf5", "w")

        # Write model to db_dir
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    @property
    def model(self):
        """Get model. Protects the model from being changed once passed to sampler."""
        return self._model

    def write_samples(self, key, data):
        """Write data to database file.
        """
        # Get the dataset
        ds = self.File[key]

        # Resize the dataset for the new samples
        old_dims = ds.shape
        new_dims = (old_dims[0] + data.shape[0], data.shape[1])
        ds.resize(new_dims)

        # Add the new samples
        ds[old_dims[0]:new_dims[0], :] = data

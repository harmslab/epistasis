from . mapping import EpistasisMap
import h5py

class EpistasisBayesianMapping(EpistasisMap):
    """Mapping object for epistatic coefficients calculated by Bayesian methods.

    Many samples are usually needed to sample Bayes theorem sufficiently. This
    can quickly become expensive in memory.
    """
    def __init__(self, fname):
        self.filename = fname+".hdf5"
        self.File = self.h5py.File(self.filename, "a")

    @property
    def log_prob(self):
        """
        """
        


    @property
    def values(self):
        """Get the values of the interaction in the system"""


    @property
    def stdeviations(self):
        """Get standard deviations from model"""
        raise Exception("not working yet.")
        return self._stdeviations

    @samples.setter
    def samples(self, samples):
        """Store a set of epistatic coefficients for a given epistasis model.
        """
        try:
            # Resize the samples dataset in the hdf5 file.
            dims = self.samples.shape
            dims_to_add = values.shape
            new_dims = (dims[0] + dims_to_add[0],) + dims[1:]
            self._samples.resize(new_dims)

            # Add the samples to the dataset.
            self._samples[dims[0]:new_dim[0],:,:] = samples

        # If samples does not exist, create it. Stores the data as a chunked dataset
        except ValueError:
            self._samples = self.File.create_dataset("samples", samples,
                chunks=True, maxshape=(None,None,None))

    @probabilities.setter
    def probabilities(self, probabilities):
        """Store a set of probabilities for a given epistasis model.
        """
        try:
            # Resize the samples dataset in the hdf5 file.
            dims = self.samples.shape
            dims_to_add = values.shape
            new_dims = (dims[0] + dims_to_add[0],) + dims[1:]
            self._probabilities.resize(new_dims)

            # Add the samples to the dataset.
            self._probabilities[dims[0]:new_dim[0],:,:] = probabilities

        # If samples does not exist, create it. Stores the data as a chunked dataset
        except ValueError:
            self._probabilities = self.File.create_dataset("probabilities", probabilities,
                chunks=True, maxshape=(None,None,None))


    @stdeviations.setter
    def stdeviations(self, stdeviations):
        """Set the standard deviations of the epistatic coefficients."""
        raise Exception("not working yet.")
        self._stdeviations = stdeviations
        #self.std = gpmap.errors.StandardDeviationMap(self)
        #self.err = gpmap.errors.StandardErrorMap(self)

# imports from seqspace dependency
from seqspace.gpm import GenotypePhenotypeMap

# Local imports
from epistasis.mapping import EpistasisMap
from epistasis.plotting.models import EpistasisPlotting
from epistasis.decomposition import generate_dv_matrix

class BaseModel(object):
    """ This object should be used as the parent class to all epistasis models.
    Manages attachment of GenotypePhenotypeMap and EpistasisMaps to the Epistasis
    models.
    """
    @classmethod
    def from_json(cls, filename, **kwargs):
        """"""
        self = cls(**kwargs)
        self.gpm = GenotypePhenotypeMap.from_json(filename)
        return self

    @classmethod
    def from_data(cls, wildtype, genotypes, phenotypes, order=1, **kwargs):
        """ Uses a simple linear, least-squares regression to estimate epistatic
        coefficients in a genotype-phenotype map. This assumes the map is linear."""

    @classmethod
    def from_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object """
        # Grab all properties from data-structure
        self = cls(**kwargs)
        self.gpm = gpm
        return self

    @classmethod
    def from_epistasis(cls, epistasis, **kwargs):
        """ """
        self = cls(**kwargs)
        self.epistasis = epistasis
        return self

    def X_helper(self,
        genotypes=None,
        coeff_labels=None,
        order=1,
        mutations=None,
        model_type="global",
        **kwargs):
        """Helper method for constructing X matrix for regression."""
        # First check genotypes are available
        if genotypes is None:
            try:
                genotypes = self.gpm.binary.genotypes
            except AttributeError:
                raise AttributeError("genotypes must be given, because no GenotypePhenotypeMap is attached to this model.")
        # Build epistasis map
        if coeff_labels is None:
            # Mutations dictionary given? if not, build one.
            if mutations is None:
                try:
                    mutations = self.gpm.mutations
                except AttributeError:
                    mutations = extract_mutations_from_genotypes(genotypes)
            # Construct epistasis mapping
            self.epistasis = EpistasisMap.from_mutations(mutations, order)
        else:
            self.epistasis = EpistasisMap.from_labels(coeff_labels)
        # Construct the X matrix (convert to binary if necessary).
        try:
            return generate_dv_matrix(genotypes, self.epistasis.labels, model_type=model_type)
        except:
            mapping =self.gpm.map("genotypes", "binary.genotypes")
            binaries = [mapping[g] for g in genotypes]
            return generate_dv_matrix(binaries, self.epistasis.labels, model_type=model_type)

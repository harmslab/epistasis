import numpy as np
from collections import OrderedDict

class EnsembleMap(object):
    
    # ------------------------------------------------------
    # Getter methods for properties of the map
    # ------------------------------------------------------
    
    @property
    def N(self):
        """ Get the number of spaces in ensemble (i.e. size of ensemble)."""
        return self._N
    
    @property
    def n(self):
        """ Get number of genotypes, i.e. size of the system. """
        return self._n
    
    @property
    def model(self):
        """ Get the type of epistasis model used in the ensemble."""
        return self._model
    
    @property
    def genotypes(self):
        """ Get the genotypes of the system. """
        return self._genotypes
        
    @property
    def phenotypes(self):
        """ Get the phenotypes of the system. """
        return self._phenotypes
    
    @property
    def phenotype_errors(self):
        """ Get the phenotypes' errors in the system. """
        return self._phenotype_errors
    
    @property
    def ensemble_wildtypes(self):
        """ Get to ordered list of wildtype states used in ensemble calculations. """
        return self._ensemble_wildtypes
        
    @property
    def interaction_genotypes(self):
        """ Get interaction genotypes."""
        return list(self._ensemble.keys())
        
    @property
    def ensemble(self):
        """ Get the values each interaction in ensemble. """
        return self._ensemble
        
    # -------------------------------------------------------
    # Setter methods for the map.
    # -------------------------------------------------------
    
    @N.setter
    def N(self, N):
        """ Set size of the ensemble. """
        try:
            self._N += N
        except AttributeError:
            self._N = N
    
    @model.setter
    def model(self, model):
        """ Set model of the ensemble. """
        self._model = model
    
    @genotypes.setter
    def genotypes(self, genotypes):
        """ Set genotypes in ensemble. """
        genotypes = sorted(genotypes)
        self._n = len(genotypes)
        self._genotypes = np.array(genotypes)
                
    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """ Set phenotypes from ordered list of phenotypes. 
            
            __Arguments__:
            
            `phenotypes` [array-like or dict] : if array-like, it musted be 
                ordered by genotype; if dict, this method automatically orders 
                the phenotypes into numpy array.
        """
        if type(phenotypes) is dict:
            self._phenotypes = self._if_dict(phenotypes)
        else:
            if len(phenotypes) != len(self._genotypes):
                raise("Number of phenotypes does not equal number of genotypes.")
            else:
                self._phenotypes = phenotypes
                
    @phenotype_errors.setter
    def phenotype_errors(self, errors):
        """ Set error from ordered list of phenotype error. 
            
            __Arguments__:
            
            `error` [array-like or dict] : if array-like, it musted be ordered by 
                genotype; if dict, this method automatically orders the errors 
                into numpy array.
        """
        if type(errors) is dict:
            self._phenotype_errors = self._if_dict(phenotype_errors)
        else:
            self._phenotype_errors = errors
            
    @ensemble_wildtypes.setter
    def ensemble_wildtypes(self, wildtypes):
        """ Set or append to the list of enemble reference (wildtype) states. """
        # If wildtypes exists, append to it; else, set it.
        try:
            self._wildtypes.append(wildtypes)
        except:
            self._wildtypes = wildtypes
            
    @ensemble.setter
    def ensemble(self, ensemble):
        """ Create or add to an ensemble map. 
            
            Arg:
            ---
            ensemble: dict
                Dictionary of mutation genotypes to their epistastic value.
        """            
        # If the ensemble has already been started, add to it; else, 
        # create a new ensemble map
        try:
            for key, value in ensemble.items():
                if key in self._ensemble:
                    self._ensemble[key].append(value)
                else:
                    self._ensemble[key] = [value]
        except AttributeError:
            self._ensemble = dict()
            for key, value in ensemble.items():
                self._ensemble[key] = [value]
                    
    # -------------------------------------------------------
    # Useful methods for the map.
    # -------------------------------------------------------

    def _map(self, keys, values):
        """ Return ordered dictionary mapping two properties in self. """
        return OrderedDict([(keys[i], values[i]) for i in range(self._n)])
        
    def _if_dict(self, dictionary):
        """ If setter method is passed a dictionary with genotypes as keys, 
            use those keys to populate array of elements in order
        """
        elements = np.empty(self._n, dtype=float)
        for i in range(self._n):
            elements[i] = dictionary[self._genotypes[i]]
        return elements
        
    def _label_to_genotype(self, label):
        """ Convert a label to its genotype representation. """
        genotype = ""
        for l in label:
            # Labels are offset by 1, remove offset for wildtype/mutation array index
            array_index = l - 1
            mutation = self.wildtype[array_index] + str(l) + self.mutations[array_index]
            genotype += mutation + ','
        # Return genotype without the last comma
        return genotype[:-1]
        
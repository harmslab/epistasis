import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from .regression import EpistasisRegression

# ------------------------------------------------------------
# Standard Machine learning curves applied to epistasis models
# ------------------------------------------------------------

class MachineLearnEpistasis(object):
    
    def __init__(self, geno_pheno_map, model, num_samples=5, test_size=.4, truncation=None):
        
        self.gpm = geno_pheno_map
        self.truncation = truncation  
        if truncation is None:
            truncation = len(self.gpm.sequences[0])
        self.interaction_order = truncation
        self.num_samples = num_samples
        self.model = model
        self.test_size = test_size
        
    def bias_variance(self, order_range=range(1,5)):
        """ Create bias-variance data to check for overfitting """
        train_score = list()
        test_score = list()
        train_mse = list()
        test_mse = list()
        sequences, phenotypes = self.gpm.create_sample(self.num_samples)
        train_sequences, test_sequences, train_phenotypes, test_phenotypes = train_test_split(
            sequences, phenotypes, test_size=self.test_size)
            
        for i in order_range:
            train_regression = self.model(train_sequences, train_phenotypes, i)
            train_regression.fit()
            train_score.append(train_regression.model_score)
            train_y = train_regression.predict(train_regression.dummy_variables_matrix)
            train_mse.append(mean_squared_error(train_phenotypes, train_y))
            
            test_regression = self.model(test_sequences, test_phenotypes, i)
            test_score.append(train_regression.score(test_regression.dummy_variables_matrix, test_regression.Y_vector))
            test_y = train_regression.predict(test_regression.dummy_variables_matrix)
            test_mse.append(mean_squared_error(test_phenotypes, test_y))
            
        return train_score, test_score, train_mse, test_mse
            
    def learning_curve(self, sample_range=range(1,10)):
        """ Create a learning curve from the genotype-phenotype map by generated data from experimental values. """
        train_score = list()
        test_score = list()
        train_mse = list()
        test_mse = list()
        
        for i in sample_range:
            sequences, phenotypes = self.gpm.create_sample(i)
            train_sequences, test_sequences, train_phenotypes, test_phenotypes = train_test_split(
                sequences, phenotypes, test_size=self.test_size)

            train_regression = self.model(train_sequences, train_phenotypes, self.interaction_order)
            train_regression.fit()
            train_score.append(train_regression.model_score)
            train_y = train_regression.predict(train_regression.dummy_variables_matrix)
            train_mse.append(mean_squared_error(train_phenotypes, train_y))
        
            test_regression = self.model(test_sequences, test_phenotypes, self.interaction_order)
            test_score.append(train_regression.score(test_regression.dummy_variables_matrix, test_regression.Y_vector))
            test_y = train_regression.predict(test_regression.dummy_variables_matrix)
            test_mse.append(mean_squared_error(test_phenotypes, test_y))
        
        return train_score, test_score, train_mse, test_mse
        
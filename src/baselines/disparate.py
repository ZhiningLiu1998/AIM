import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset

class DisparateImpactRemovalClassifier(BaseEstimator):

    def __init__(self, estimator, repair_level:int=1.0, verbose=False, random_state=None):
        assert isinstance(estimator, ClassifierMixin), "estimator must be a classifier"
        assert repair_level >= 0.0 and repair_level <= 1.0, "repair_level must be in [0, 1]"
        assert isinstance(verbose, bool), "verbose must be a boolean"
        assert isinstance(random_state, int) or random_state is None, "random_state must be an integer or None"
                
        self.estimator = estimator
        self.repair_level = repair_level
        self.verbose = verbose
        self.random_state = random_state
        self.set_params(random_state=random_state)

    def set_params(self, **kwargs):
        try:
            self.estimator.set_params(**kwargs)
        except:
            pass
 
    def fit(self, X, y, sensitive_features):
        self.preprocessor = DisparateImpactRemover(
            repair_level=self.repair_level, sensitive_attribute=0
        )
        X_processed = self.remove_bias(X, sensitive_features)
        self.estimator.fit(X_processed, y)
        return self

    def remove_bias(self, X, sensitive_features):
        """Remove bias from X using the DisparateImpactRemover preprocessor."""
        assert (X[:, 0] == sensitive_features).all(), \
            "The 1st column of X must be the sensitive attribute."
        
        y_dummy = np.zeros(X.shape[0])
        aif360_data = BinaryLabelDataset(
            df=pd.DataFrame(np.hstack([X, y_dummy.reshape(-1, 1)])), 
            label_names=[X.shape[1]],
            protected_attribute_names=[0],
            privileged_protected_attributes = [[1.0]],
            unprivileged_protected_attributes = [[0.0]],
        )
        X_processed = self.preprocessor.fit_transform(aif360_data).features
        return X_processed

    def predict(self, X, sensitive_features):
        X_processed = self.remove_bias(X, sensitive_features)
        return self.estimator.predict(X_processed)

    def predict_proba(self, X, sensitive_features):
        X_processed = self.remove_bias(X, sensitive_features)
        return self.estimator.predict_proba(X_processed)
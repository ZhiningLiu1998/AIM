from sklearn.base import BaseEstimator, clone
from aif360.sklearn.preprocessing import LearnedFairRepresentations
import pandas as pd

class LFRClassifier(BaseEstimator):

    def __init__(self, estimator, random_state=None, **kwargs):
        self.estimator = estimator
        self.random_state = random_state
        self.set_params(random_state=random_state)

    def set_params(self, **kwargs):
        try:
            self.estimator.set_params(**kwargs)
        except:
            pass

    def fit(self, X_train, y_train, sensitive_features):
        # guarantee X_train 0-th column is sensitive feature
        assert (X_train[:, 0] == sensitive_features).all()

        lfr = LearnedFairRepresentations(
            prot_attr=X_train[:, 0],
            random_state=self.random_state
        )
        lfr.fit(pd.DataFrame(X_train), y_train, priv_group=1)
        self.lfr = lfr

        # transform data
        X_train_edit = lfr.transform(pd.DataFrame(X_train))

        self.clf = clone(self.estimator)
        self.clf.fit(X_train_edit, y_train)
        
        return self

    def predict(self, X_test):
        X_test_edit = self.lfr.transform(pd.DataFrame(X_test))
        return self.clf.predict(X_test_edit)

    def predict_proba(self, X_test):
        X_test_edit = self.lfr.transform(pd.DataFrame(X_test))
        return self.clf.predict_proba(X_test_edit)
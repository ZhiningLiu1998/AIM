from sklearn.base import BaseEstimator

class MimicClassifier(BaseEstimator):
    def __init__(self, estimator, random_state=None):
        self.estimator = estimator
        self.random_state = random_state
        self.estimator.set_params(random_state=random_state)

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
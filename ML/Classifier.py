import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd


class ClassifierGraspPlanner():
    """
    Grasp planner based on scikit-learn classifier.

    Uses an internal Pipeline:
        [ StandardScaler -> RandomForestClassifier ]
    Can replace it with SVC / MLPClassifier / XGBoost etc.
    """

    def __init__(
        self,
        classifier: Optional[BaseEstimator] = None,
    ) -> None:
        # Default to a simple RandomForestClassifier
        if classifier is None:
            classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )

        # Using Pipeline for easy addition of standardization, feature processing, etc.
        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ]
        )
        self._is_trained: bool = False

    # ------------ Implement abstract methods ------------

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classification model"""
        self.pipeline.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary classification results (0/1)"""
        self._check_trained()
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability for each class (column 1 is success probability)"""
        self._check_trained()
        
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            return self.pipeline.predict_proba(X)
        else:
            # Use decision_function to estimate probabilities
            decision = self.pipeline.decision_function(X)
            min_d, max_d = decision.min(), decision.max()
            if max_d - min_d < 1e-8:
                return np.full((len(decision), 2), 0.5)
            prob_success = (decision - min_d) / (max_d - min_d)
            prob_fail = 1.0 - prob_success
            return np.vstack([prob_fail, prob_success]).T

    def save(self, path: str) -> None:
        """Save the pipeline to disk"""
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        """Load the pipeline from disk"""
        self.pipeline = joblib.load(path)
        self._is_trained = True

    # ------------ Internal Tools ------------

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("ClassifierGraspPlanner is not trained yet.")

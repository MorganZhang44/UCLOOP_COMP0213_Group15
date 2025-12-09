from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split


@dataclass
class GraspDataset:
    """
    Encapsulates the grasp dataset:
        - pandas DataFrame
        - feature_columns: List of input feature column names (pose parameters)
        - label_column: Label column name (0: fail, 1: success)
    """
    df: pd.DataFrame
    feature_columns: List[str]
    label_column: str

    @classmethod
    def from_csv(
        cls,
        path: str,
        label_column: str,
        feature_columns: Optional[List[str]] = None,
    ) -> "GraspDataset":
        """Load dataset from a CSV file"""
        df = pd.read_csv(path)

        if feature_columns is None:
            # Assume the last column is the label, and all previous columns are features
            all_cols = list(df.columns)
            if label_column not in all_cols:
                raise ValueError(
                    f"Label column '{label_column}' not found in CSV columns: {all_cols}"
                )
            feature_columns = [c for c in all_cols if c != label_column]

        return cls(df=df, feature_columns=feature_columns, label_column=label_column)

    
    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into training and validation sets.

        Returns:
            X_train, X_val, y_train, y_val
        """
        X = self.df[self.feature_columns].values
        y = self.df[self.label_column].values

        if stratify:
            stratify_y = y
        else:
            stratify_y = None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
        return X_train, X_val, y_train, y_val
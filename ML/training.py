from typing import Tuple, Optional, List
from sklearn.metrics import accuracy_score, classification_report
from ML.GraspDataset import GraspDataset
from ML.Classifier import ClassifierGraspPlanner


def train_classifier_based_planner(
    dataset_path: str,
    model_output_path: str,
    label_column: str = "label",
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
) -> ClassifierGraspPlanner:
    """
    Train a classifier-based grasp planner and print validation set performance.

    Parameters:
        dataset_path: Path to the CSV or pickle file
        label_column: Name of the label column (0/1)
        feature_columns: List of feature column names (if None, defaults to "last column is label, others are features")
        test_size: Proportion of the validation set
        model_output_path: Path to save the model

    Returns:
        A trained ClassifierGraspPlanner instance
    """

    # 1. Load the dataset

    ds = GraspDataset.from_csv(
            dataset_path,
            label_column=label_column,
            feature_columns=feature_columns,
        )

    print("Feature columns:", ds.feature_columns)
    print("Label column:", ds.label_column)
    print("Dataset size:", len(ds.df))

    # 2. Split the training/validation set
    X_train, X_val, y_train, y_val = ds.train_test_split(test_size=test_size)

    # 3. Initialize the planner (internally creates the classifier)
    planner = ClassifierGraspPlanner()

    # 4. Train
    planner.train(X_train, y_train)

    # 5. Evaluate on the validation set
    y_pred = planner.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc * 100:.2f}%")
    print("Classification report:")
    print(classification_report(y_val, y_pred, digits=4))

    # 6. Save the model to a file (for easy loading during Testing Phase (b) in simulation)
    planner.save(model_output_path)
    print(f"Trained model saved to: {model_output_path}")

    return planner
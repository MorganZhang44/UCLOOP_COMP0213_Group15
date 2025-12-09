# 🤖 Learning-Based Grasp Classification Using PyBullet Simulation

This project implements a complete pipeline for data-driven grasp success prediction using PyBullet simulation and machine learning.
The system generates randomized gripper poses, executes grasp trials in simulation, labels outcomes automatically, trains a classifier to distinguish successful grasps, and evaluates the trained model on new simulated samples.

## 1️⃣ Project Overview

The project provides three main functionalities:

- **➡️ Dataset Generation:**
    - Randomly sample gripper poses, execute a grasp in PyBullet, and automatically label it as success/failure.

- **➡️ Classifier Training:**
    - Train a machine learning model (RandomForest-based) using the collected dataset of poses and labels.

- **➡️ Planner Testing:**
    - Evaluate the learned classifier by predicting grasp success for newly simulated trials.

All three modes are integrated into main.py through argparse, allowing the user to run each part from the command line.

## 2️⃣ Installation

**Requirements:**

- Python 3.8+
- PyBullet
- NumPy
- Pandas
- scikit-learn
- joblib

**Install all dependencies with:**
```bash
pip install -r requirements.txt
```
No additional configuration files are needed; all parameters are provided using command-line arguments.

## 3️⃣ How to Run

The project supports three execution modes:
generator, training, and testing.
These are selected using the --mode argument.

#### 🦾 (A) Generate Dataset

This mode collects grasp samples and saves them into a CSV file.

```bash
python main.py generator \
    --gripper 2f \
    --object cube \
    --num 150 \
    --output data/cube_150.csv
```

| Argument | Description |
|-----------|-------------|
| gripper (2f,3f) | Choose two-finger or three-finger gripper |
| object (cube,cylinder) | Object type in the scene |
| num | Number of samples to generate |
| output | Output CSV file path |

#### 🚀 (B) Train Classifier

Train a RandomForest-based classifier using a generated dataset.

```bash
python main.py training \
    --dataset data/cube_150.csv \
    --model model/cube_rf.joblib
```

| Argument | Description |
|-----------|-------------|
| dataset | Path to the input CSV dataset |
| model | Where to save the trained model |
| test_size | Ratio used for validation split (default: 0.2) |

The script prints validation accuracy and a classification report.

#### ⚙️ (C) Test the Planner

Load a trained model and evaluate its predictions in simulation.

```bash
python main.py testing \
    --gripper 2f \
    --object cube \
    --num 50 \
    --model model/cube_rf.joblib
```

| Argument | Description |
|-----------|-------------|
| gripper  | Gripper type used in simulation |
| object  | Object to grasp |
| num | Number of test samples |
| model | Path to the trained model |

The script prints prediction vs. ground truth for each trial and reports total accuracy.

## 4️⃣ Directory Structure
```bash
CourseWork/
│── main.py                # Entry point (generator / training / testing)
│── Env/
│   └── SimEnv.py          # PyBullet simulation environment
│── gripper/
│   ├── Base_pawl.py       # Base gripper class
│   ├── pawl_2f.py         # Two-finger gripper
│   └── pawl_3f.py         # Three-finger gripper
│── object/
│   ├── object.py          # Base object class
│   ├── cube.py            # Cube object
│   └── cylinder.py        # Cylinder object
│── algorithm/
│   └── random_gripper.py  # Random pose generation
│── ML/
│   ├── GraspDataset.py    # Dataset loader and splitter
│   ├── Classifier.py      # RandomForest grasp classifier
│   └── training.py        # Training pipeline
│── requirements.txt
```

## 5️⃣ Notes

1. All parameters are passed through command-line arguments; no external config file is used.

2. PyBullet GUI will open during simulation modes (generator and testing).

3. Generated datasets and saved ML models should be stored under data/ and model/, respectively

# Dataset Documentation
## Human Activity Recognition with Smartphones (UCI HAR)

## 1. Dataset Overview

This project uses the **Human Activity Recognition with Smartphones** dataset. It is a supervised multi-class classification dataset containing measurements collected from smartphone inertial sensors while volunteers performed everyday activities.

- **Kaggle dataset:** https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
- **Original UCI dataset:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Task:** Predict the activity being performed from a 561-dimensional sensor-feature vector.
- **Dataset type:** Multivariate sensor data represented by engineered numerical features.

The original experiment involved 30 volunteers aged 19–48. Each participant wore a Samsung Galaxy S II smartphone at the waist while performing six activities. The phone's accelerometer and gyroscope recorded movement signals.

## 2. Data Collection

| Item | Description |
|---|---|
| Participants | 30 volunteers |
| Device | Samsung Galaxy S II |
| Device position | At the waist |
| Sensors | Accelerometer and gyroscope |
| Accelerometer | Three axes (X, Y, Z) |
| Gyroscope | Three axes (X, Y, Z) |
| Sampling rate | 50 Hz |
| Window duration | 2.56 seconds |
| Readings per window | 128 |
| Window overlap | 50% |
| Classification labels | 6 activities |

The original sensor signals were filtered and divided into fixed-width sliding windows. Acceleration was separated into body acceleration and gravity components. Features were then calculated in both the time and frequency domains.

## 3. Dataset Size and Files

The version used in this project is provided as two CSV files:

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `train.csv` | 7,352 | 563 | Model development and training |
| `test.csv` | 2,947 | 563 | Final evaluation |
| **Total** | **10,299** | **563** | Combined dataset |

Each CSV contains:

- 561 numerical feature columns.
- One `Activity` column containing the activity label.
- One `subject` column identifying the participant.

The 70%/30% participant-based training/test partition is part of the original dataset. In the project notebook, a further group-based 80%/20% training/validation split is created from `train.csv`.

## 4. Features

### 4.1 Input Features

The model uses 561 engineered numerical features derived from accelerometer and gyroscope signals. They represent measurements and statistical characteristics in the time and frequency domains.

The original dataset feature set includes characteristics such as:

- Mean and standard deviation.
- Signal magnitude and energy-related measurements.
- Signal entropy and other statistical measures.
- Correlation-related measurements.
- Frequency-domain characteristics derived from transformed signals.

The feature vector includes information derived from body acceleration, total acceleration, gravity-related acceleration, and gyroscope measurements. The exact feature names are stored in the CSV column headers and are retained by the notebook in `feature_columns`.

**Input shape expected by the trained model:** `(number_of_samples, 561)`.

The model uses the engineered 561-column feature representation. The Streamlit demo does not accept raw 128-step sensor sequences.

### 4.2 Target Label: `Activity`

The target column contains one of six activity names:

| Activity label | Meaning |
|---|---|
| `WALKING` | Walking |
| `WALKING_UPSTAIRS` | Walking upstairs |
| `WALKING_DOWNSTAIRS` | Walking downstairs |
| `SITTING` | Sitting |
| `STANDING` | Standing |
| `LAYING` | Lying down |

`LabelEncoder` converts these text labels into integer class IDs for model training. The fitted encoder is saved and used by the demo to convert predictions back to readable activity names.

### 4.3 Participant Identifier: `subject`

The `subject` column identifies the volunteer associated with each sample. It is used for group-based splitting and leakage checks; it is **not** included among the 561 model input features.

## 5. Training, Validation, and Testing

The notebook uses the provided official `train.csv` and `test.csv` files.

### 5.1 Training and Validation Split

- `train.csv` is separated into input features (`X`), target labels (`y`), and participant groups (`subject`).
- `GroupShuffleSplit` creates one split with `test_size=0.20` and `random_state=42`.
- The `subject` identifier is used as the grouping variable.
- The notebook checks that training and validation participant IDs do not overlap.

This group-based split is intended to prevent samples from the same participant from appearing in both the training and validation subsets.

### 5.2 Official Test Set

The provided `test.csv` is held out for final evaluation. The notebook checks that the feature columns in training and test data match.

For the final exported model, the notebook fits a new `StandardScaler` on all feature rows in the official training file, trains the final model using the full official training set, and evaluates it on the official test set.

### 5.3 Split Summary

| Subset | Source | Use |
|---|---|---|
| Training subset | 80% group split from `train.csv` | Model fitting and experiments |
| Validation subset | 20% group split from `train.csv` | Model selection and experiment evaluation |
| Final training data | All rows in `train.csv` | Training the exported final model |
| Test set | `test.csv` | Final performance evaluation |

The notebook determines the exact training and validation row counts from the selected participant groups. The official file sizes are 7,352 training rows and 2,947 test rows.

## 6. Preprocessing

The notebook applies the following preprocessing steps:

1. **Load CSV files:** Read `train.csv` and `test.csv` using pandas.
2. **Separate columns:** Remove `Activity` and `subject` from the input feature matrix. Keep `Activity` as the target and `subject` as the grouping identifier.
3. **Check feature consistency:** Confirm that training and test feature columns are identical and in the same order.
4. **Encode labels:** Apply `LabelEncoder` to convert activity names into integer labels.
5. **Group-based split:** Split the official training data into training and validation groups using `subject`.
6. **Feature scaling:** Apply `StandardScaler`. During model-comparison experiments, the scaler is fitted only on the training subset and then applied to validation and test features.
7. **Tensor conversion:** Convert feature arrays to `float32` PyTorch tensors and labels to integer (`long`) tensors.

### Additional Experiments

The notebook also investigates:

- Gaussian noise augmentation on the training features, using a noise standard deviation of `0.05`.
- Self-supervised masked-feature reconstruction pretraining.
- Multiple random seeds and learning rates.
- An ablation comparison between the advanced model with and without feature attention.

These are experimental strategies, not additional mandatory preprocessing steps for the final Streamlit inference pipeline.

## 7. Missing Values and Data Quality

The notebook inspects missing values, duplicated rows, activity distribution, subject IDs, and feature dimensions during data exploration. It also checks for subject overlap between training and validation groups.

For the Streamlit demo, uploaded CSV data is checked for:

- Missing required feature columns.
- Empty files.
- Missing or infinite numerical values.

The app expects the same 561 feature columns used during training. It does not automatically create features from raw sensor readings.

## 8. Dataset Use in the Project

The dataset supports the project's objective of classifying human physical activity from smartphone inertial sensor measurements. The 561-feature representation is passed to the Random Forest baseline and the deep-learning architectures described in `README.md`.

The final deployed model returns one predicted activity label and a confidence value for each uploaded sample. It can also display the predicted probability for each of the six activity classes.

## 9. Dataset References

1. UCI Machine Learning Repository. *Human Activity Recognition Using Smartphones*. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Kaggle. *Human Activity Recognition with Smartphones*. https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
3. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN.

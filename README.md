# Human Activity Recognition (HAR) Using Smartphone Sensors

## Project Overview

This project uses machine learning and deep learning to recognize six human activities from smartphone sensor features. It is based on the **Human Activity Recognition with Smartphones** dataset.

The dataset contains 561 pre-engineered numerical features for each activity window. The project trains and compares several classifiers, selects a final neural-network model, evaluates it on the official test set, and provides a Streamlit interface for batch predictions from a CSV file.

## Activities

The model predicts one of the following classes:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

## Dataset

- **Source:** [Kaggle – Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
- **Original source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Participants:** 30 volunteers
- **Sensor device:** Samsung Galaxy S II, worn at the waist
- **Sensors:** 3-axis accelerometer and 3-axis gyroscope
- **Sampling frequency:** 50 Hz
- **Window length:** 2.56 seconds (128 readings), with 50% overlap
- **Features:** 561 time-domain and frequency-domain features
- **Training records:** 7,352
- **Test records:** 2,947
- **Total records:** 10,299

The CSV files used by the notebook are `train.csv` and `test.csv`. Each contains 561 feature columns plus `Activity` (the target label) and `subject` (the participant identifier), for 563 columns in total.

See [Dataset_Documentation.md](Dataset_Documentation.md) for the detailed data description and preprocessing workflow.

## Models and Experiments

The notebook implements and compares:

1. **Random Forest baseline** – 200 trees.
2. **1D CNN** – convolutional layers for learning patterns across the ordered feature vector.
3. **Feature Transformer** – a Transformer encoder with feature projection and positional embeddings.
4. **Advanced HAR model** – learned feature attention followed by a fully connected residual block and classifier.
5. **Advanced HAR without attention** – an ablation variant used to measure the effect of removing the feature-attention module.
6. **Self-supervised pretraining experiment** – a masked-feature Transformer pretrained to reconstruct masked feature values, followed by transfer learning and fine-tuning.

The experiments also investigate random seeds, learning rates, Gaussian-noise augmentation, and the effect of feature attention.

## Methodology

1. Load the provided training and test CSV files.
2. Separate the 561 input features from `Activity` and `subject`.
3. Encode the six activity labels using `LabelEncoder`.
4. Create an 80/20 group-based training/validation split from the provided training data using `GroupShuffleSplit`, grouping by `subject` to avoid participant overlap between these two subsets.
5. Fit `StandardScaler` on the training subset and apply it to validation and test data.
6. Train and compare the baseline and deep-learning models.
7. Retrain the selected deployable model on the full official training set.
8. Evaluate the final model on the official test set.
9. Save the model weights and preprocessing artifacts for the Streamlit demo.

## Final Model and Test Results

The saved demo model is `AdvancedHARWithoutAttention`, a feed-forward neural network with:

- Input dimension: 561
- Hidden dimension: 256
- Batch normalization and GELU activation
- A residual block with dropout (0.3)
- A classifier with 128 hidden units
- Output dimension: 6 classes

The final model was trained for 40 epochs using Adam with a learning rate of 0.001 and seed 42.

**Official test-set results recorded in the notebook:**

| Metric | Result |
|---|---:|
| Accuracy | 95.83% |
| Macro F1-score | 95.81% |
| Test samples | 2,947 |
| Misclassified samples | 123 |

These results are from the final saved model evaluated on the official test set. They should not be confused with the validation results reported during model-comparison experiments.

## Demo Application

The Streamlit application (`app.py`) accepts a CSV file containing the same 561 feature columns used during training. It then:

1. Checks that all required feature columns are present.
2. Checks for missing or infinite feature values.
3. Applies the saved `StandardScaler`.
4. Loads the saved PyTorch model and predicts an activity for each row.
5. Displays the predicted activity, confidence percentage, and class-probability chart for a selected sample.

**Important:** The demo expects preprocessed feature-vector CSV input. It does not currently process raw accelerometer/gyroscope time-series readings directly.

### Required project structure

Place the files in the following structure:

```text
project/
├── app.py
└── HAR_Demo/
    ├── har_model.pth
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── metadata.json
```

The four files inside `HAR_Demo/` are generated by the final export cell in the notebook. Keep the model architecture in `app.py` consistent with the architecture used to create `har_model.pth`.

### Installation

Use Python 3.10 or a compatible Python version and install the required packages:

```bash
pip install streamlit torch numpy pandas scikit-learn joblib
```

### Run the application

From the project directory, run:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser, then upload a CSV containing the 561 expected feature columns.

## Repository Files

- `README.md` – project overview, methodology, models, results, setup, and demo instructions.
- `Dataset_Documentation.md` – dataset source, collection details, features, labels, file structure, and preprocessing.
- `deeeeep33.ipynb` – notebook containing data exploration, preprocessing, model experiments, evaluation, and artifact export.
- `app.py` – Streamlit application for loading the saved model and predicting activities from uploaded feature CSV files.
- `HAR_Demo/` – exported model and preprocessing artifacts required by the demo (generated by the notebook).

## Reproducibility Notes

- The notebook uses random seeds `21`, `42`, and `2026` for repeated model comparisons.
- The final exported model uses seed `42`.
- The notebook selects CUDA when available and otherwise runs on CPU.
- The notebook's dataset path is configured for Kaggle. If running it elsewhere, update `DATA_PATH` to the folder containing `train.csv` and `test.csv`.

## References

1. [Kaggle – Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
2. [UCI Machine Learning Repository – Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
3. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN.

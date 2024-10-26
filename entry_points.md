# Entry Points

This document describes the main entry points of the project and their functions. For detailed usage instructions, please refer to the "Usage" section in [README.md](./README.md).

## 1. Data Preparation:
- **Description**: This script downloads the raw training data, performs preprocessing, and outputs cleaned data.
- **Inputs**: 
  - `RAW_DATA_DIR` (specified in `SETTINGS.json`)
- **Outputs**:
  - `TRAIN_DATA_CLEAN_DIR` (specified in `SETTINGS.json`)
- **Detailed Instructions**: See the "Data Download and Preprocessing" subsection in the "Usage" section of README.md.

## 2. Model Training:
- **Description**: This script trains the model using the cleaned training data.
- **Inputs**:
  - `TRAIN_DATA_CLEAN_PATH` (specified in `SETTINGS.json`)
- **Outputs**:
  - `MODEL_CHECKPOINT_DIR` (specified in `SETTINGS.json`)
- **Detailed Instructions**: See the "Train Keypoint Detection Models" and "Train Classification Models" subsections in the "Usage" section of README.md.

## 3. Model Prediction:
- **Description**: This script loads a trained model and makes predictions on new samples.
- **Inputs**:
  - `RAW_DATA_DIR` (specified in `SETTINGS.json`)
  - `MODEL_CHECKPOINT_DIR` (specified in `SETTINGS.json`)
- **Outputs**:
  - `SUBMISSION_DIR` (specified in `SETTINGS.json`)
- **Detailed Instructions**: See the "Prediction" subsection in the "Usage" section of README.md.

# RSNA 2024 Lumbar Spine Degenerative Classification - 3rd Place Solution

This is the 3rd place solution for the RSNA 2024 Lumbar Spine Degenerative Classification competition. For more details, please refer to the [discussion](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539453).

## Environment

This project was developed in the following environment:

### Workstation

- **CPU**: AMD EPYC (Rome)
- **Memory**: 120 GB RAM
- **GPU**: NVIDIA A100 (48 GB VRAM)
- **OS**: Ubuntu 20.04.6 (5.4.0-181-generic)

### Laptop

- **CPU**: AMD Ryzen 7 5700X 8-Core Processor
- **Memory**: 64 GB RAM
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM)
- **OS**: Ubuntu 20.04.6 (5.15.0-113-generic)

**Note**: Some models in this project require 48 GB of VRAM to run. Ensure that you have access to a GPU with at least 48 GB of VRAM, such as the NVIDIA A100, when executing these models.

## Prerequisites

Before running this project, make sure the following are already installed on your system:

- **[NVIDIA Driver](https://www.nvidia.com/en-us/drivers/)**
- **[CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)**
- **[Docker](https://docs.docker.com/engine/install/debian/)**  
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**  
- **[Kaggle API](https://www.kaggle.com/docs/api)**

The Kaggle API must be set up with your API token located at `~/.kaggle/kaggle.json`. Refer to Kaggle API documentation for instructions on generating your API token.

## Setup

This project is set up to run inside a Docker container.

1. Clone the Repository

   ```bash
   git clone https://github.com/Moyasii/Kaggle-2024-RSNA-Pub.git
   ```

2. Move into the Project Directory

   ```bash
   cd Kaggle-2024-RSNA-Pub
   ```

3. Build the Docker Image

   ```bash
   docker build -t rsna2024_3rd_img .
   ```

4. Run the Docker Container

   ```bash
   docker run --gpus all -it --rm --name rsna2024_3rd_cont --shm-size 24G -v $(pwd):/kaggle -v ~/.kaggle:/root/.kaggle rsna2024_3rd_img /bin/bash
   ```

## Usage

This project can be used in two ways: quick prediction using pretrained models, or a full workflow that includes data preparation, model training, and prediction.

### Quick Predict with Pretrained Models

You can quickly run predictions on test data using pretrained models by executing the following command:

```bash
scripts/run_quick.sh
```

This script automatically downloads the pretrained models from the [Kaggle Dataset](https://www.kaggle.com/datasets/akinosora/rsna2024-lsdc-3rd-models-pub) during execution, producing results in the shortest time.

### Full Training and Prediction Workflow

You can execute all steps at once or run each step individually.

#### All at Once

To run the entire workflow, including data preparation, model training, and prediction, use the following script:

```bash
scripts/run.sh
```

This script sequentially performs all necessary steps to complete the workflow.

#### Step by Step

If you prefer to run each step individually, follow the instructions below.

1. Data Download and Preprocessing

   Prepare the dataset in the `<RAW_DATA_DIR>` directory and run:

   ```bash
   python3 src/utils/prepare_data.py
   python3 src/utils/generate_kfold.py
   ```

2. Train Keypoint Detection Models

   Start by creating the datasets and training the keypoint detection models.

   ```bash
   ### Moyashii
   # Create sagittal dataset v2 and train the sagittal keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v2.py moyashii 2024
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v2_0008_config.yaml 0008 --dst_root stage1/moyashii/v2/0008
   # Create sagittal dataset v4 and train the sagittal keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v4.py moyashii 2024
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v4_0005_config.yaml 0005 --dst_root stage1/moyashii/v4/0005
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v4_0012_config.yaml 0012 --dst_root stage1/moyashii/v4/0012
   # Create sagittal dataset v6 and train the sagittal keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v6.py moyashii 2024
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v6_0002_config.yaml 0002 --dst_root stage1/moyashii/v6/0002
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v6_0003_config.yaml 0003 --dst_root stage1/moyashii/v6/0003
   # Create axial dataset v5 and train the axial keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v5.py moyashii 2024
   python3 src/stage1/moyashii/axial/train.py src/stage1/moyashii/axial/config/v5_0003_config.yaml 0003 --dst_root stage1/moyashii/v5/0003

   ### tkmn
   # Create sagittal dataset v2 and train the sagittal keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v2.py tkmn 42
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v2_0008_config.yaml 0008 --dst_root stage1/tkmn/v2/0008
   # Create sagittal dataset v4 and train the sagittal keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v4.py tkmn 42
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v4_0005_config.yaml 0005 --dst_root stage1/tkmn/v4/0005
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v4_0012_config.yaml 0012 --dst_root stage1/tkmn/v4/0012
   # Create sagittal dataset v6 and train the sagittal keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v6.py tkmn 42
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v6_0002_config.yaml 0002 --dst_root stage1/tkmn/v6/0002
   python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v6_0003_config.yaml 0003 --dst_root stage1/tkmn/v6/0003
   # Create axial dataset v5 and train the axial keypoint models
   python3 src/stage1/moyashii/tools/create_dataset_v5.py tkmn 42
   python3 src/stage1/moyashii/axial/train.py src/stage1/tkmn/axial/config/v5_0003_config.yaml 0003 --dst_root stage1/tkmn/v5/0003
   ```

3. Train Classification Models

   Use the trained keypoint detection models to create datasets for classification and train the classification models.

   ```bash
   ### tkmn
   # Create the dataset using keypoint detection models
   python3 src/stage2/moyashii/tools/create_dastaset_v9.py tkmn
   # Train the center classification models (using dataset v9)
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0022.yaml S0022 --dst_root stage2/tkmn/center/v9/0022 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0046.yaml S0046 --dst_root stage2/tkmn/center/v9/0046 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0051.yaml S0051 --dst_root stage2/tkmn/center/v9/0051 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0056.yaml S0056 --dst_root stage2/tkmn/center/v9/0056 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0078.yaml S0078 --dst_root stage2/tkmn/center/v9/0078 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0123.yaml S0123 --dst_root stage2/tkmn/center/v9/0123 --options folds=[0,1]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0141.yaml S0141 --dst_root stage2/tkmn/center/v9/0141 --options folds=[1,2]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0143.yaml S0143 --dst_root stage2/tkmn/center/v9/0143 --options folds=[2]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0162.yaml S0162 --dst_root stage2/tkmn/center/v9/0162 --options folds=[3]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0188.yaml S0188 --dst_root stage2/tkmn/center/v9/0188 --options folds=[0,1]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0189.yaml S0189 --dst_root stage2/tkmn/center/v9/0189 --options folds=[0,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0202.yaml S0202 --dst_root stage2/tkmn/center/v9/0202 --options folds=[2]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0204.yaml S0204 --dst_root stage2/tkmn/center/v9/0204 --options folds=[2,3]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0211.yaml S0211 --dst_root stage2/tkmn/center/v9/0211 --options folds=[3,4]
   # Train the side classification models (using dataset v9)
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0024.yaml 0024 --dst_root stage2/tkmn/side/v9/0024 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0057.yaml 0057 --dst_root stage2/tkmn/side/v9/0057 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0068.yaml 0068 --dst_root stage2/tkmn/side/v9/0068 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0093.yaml 0093 --dst_root stage2/tkmn/side/v9/0093 --options folds=[0,1,2,3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0101.yaml 0101 --dst_root stage2/tkmn/side/v9/0101 --options folds=[0,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0123.yaml 0123 --dst_root stage2/tkmn/side/v9/0123 --options folds=[0,1]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0124.yaml 0124 --dst_root stage2/tkmn/side/v9/0124 --options folds=[2]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0138.yaml 0138 --dst_root stage2/tkmn/side/v9/0138 --options folds=[0,1]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0139.yaml 0139 --dst_root stage2/tkmn/side/v9/0139 --options folds=[3,4]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0145.yaml 0145 --dst_root stage2/tkmn/side/v9/0145 --options folds=[1,2]
   python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0203.yaml 0203 --dst_root stage2/tkmn/side/v9/0203 --options folds=[1]

   ### Moyashii
   # Create the dataset using keypoint detection models
   python3 src/stage2/moyashii/tools/create_dastaset_v8.py moyashii
   python3 src/stage2/moyashii/tools/create_dastaset_v9.py moyashii
   python3 src/stage2/moyashii/tools/create_dastaset_v11.py moyashii
   # Train the center classification models (using dataset v9)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0053.yaml 0053 --dst_root stage2/moyashii/center/v9/0053_42 --options folds=[0,1,2,3,4] seed=42
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0053.yaml 0053 --dst_root stage2/moyashii/center/v9/0053_2024 --options folds=[0,1,2,3,4] seed=2024
   # Train the side classification models (using dataset v9)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0054.yaml 0054 --dst_root stage2/moyashii/side/v9/0054_42 --options folds=[0,1,2,3,4] seed=42
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0054.yaml 0054 --dst_root stage2/moyashii/side/v9/0054_2024 --options folds=[0,1,2,3,4] seed=2024
   # Train the center classification models (using dataset v11)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0057.yaml 0057 --dst_root stage2/moyashii/center/v11/0057_42 --options folds=[0,1,2,3,4] seed=42
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0057.yaml 0057 --dst_root stage2/moyashii/center/v11/0057_2024 --options folds=[0,1,2,3,4] seed=2024
   # Train the side classification models (using dataset v11)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0058.yaml 0058 --dst_root stage2/moyashii/side/v11/0058_42 --options folds=[0,1,2,3,4] seed=42
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0058.yaml 0058 --dst_root stage2/moyashii/side/v11/0058_2024 --options folds=[0,1,2,3,4] seed=2024
   # Create pseudo label v1
   python3 src/utils/create_pseudo_label.py train_fold.csv train_fold_pseudo_v1.csv \
      --spinal_csvs \
      stage2/moyashii/center/v9/0053_42/best_score_submission.csv \
      stage2/moyashii/center/v9/0053_2024/best_score_submission.csv \
      stage2/moyashii/center/v11/0057_42/best_score_submission.csv \
      stage2/moyashii/center/v11/0057_2024/best_score_submission.csv \
      --lr_csvs \
      stage2/moyashii/side/v9/0054_42/best_score_submission.csv \
      stage2/moyashii/side/v9/0054_2024/best_score_submission.csv \
      stage2/moyashii/side/v11/0058_42/best_score_submission.csv \
      stage2/moyashii/side/v11/0058_2024/best_score_submission.csv
   # Train the center classification models (using dataset v11 with pseudo labels v1)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0109.yaml 0109 --dst_root stage2/moyashii/center/v11/0109 --options folds=[0,1,2,3,4]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0113.yaml 0113 --dst_root stage2/moyashii/center/v11/0113 --options folds=[0,1,2,3,4]
   # Train the side classification models (using dataset v11 with pseudo labels v1)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0074.yaml 0074 --dst_root stage2/moyashii/side/v11/0074 --options folds=[0,1,2,3,4]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0082.yaml 0082 --dst_root stage2/moyashii/side/v11/0082 --options folds=[0,1,2,3,4]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0114.yaml 0114 --dst_root stage2/moyashii/side/v11/0114 --options folds=[0,1,2,3,4]
   # Create pseudo label v3
   python3 src/utils/create_pseudo_label.py train_fold.csv train_fold_pseudo_v3.csv \
      --spinal_csvs \
      stage2/moyashii/center/v11/0109/best_score_submission.csv \
      stage2/moyashii/center/v11/0113/best_score_submission.csv \
      stage2/tkmn/center/v9/0022/best_score_submission.csv \
      stage2/tkmn/center/v9/0046/best_score_submission.csv \
      stage2/tkmn/center/v9/0051/best_score_submission.csv \
      stage2/tkmn/center/v9/0056/best_score_submission.csv \
      stage2/tkmn/center/v9/0078/best_score_submission.csv \
      --lr_csvs \
      stage2/moyashii/side/v11/0074/best_score_submission.csv \
      stage2/moyashii/side/v11/0082/best_score_submission.csv \
      stage2/moyashii/side/v11/0114/best_score_submission.csv \
      stage2/tkmn/side/v9/0024/best_score_submission.csv \
      stage2/tkmn/side/v9/0057/best_score_submission.csv \
      stage2/tkmn/side/v9/0068/best_score_submission.csv \
      stage2/tkmn/side/v9/0093/best_score_submission.csv
   # Train the center classification models (using dataset v11 with pseudo labels v3)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0152.yaml 0152 --dst_root stage2/moyashii/center/v11/0152 --options folds=[1]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0176.yaml 0176 --dst_root stage2/moyashii/center/v11/0176 --options folds=[3,4]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0188.yaml 0188 --dst_root stage2/moyashii/center/v11/0188 --options folds=[0,4]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0196.yaml 0196 --dst_root stage2/moyashii/center/v11/0196 --options folds=[2,3]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0211.yaml 0211 --dst_root stage2/moyashii/center/v11/0211 --options folds=[0,4]
   # Train the side classification models (using dataset v11 with pseudo labels v3)
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0157.yaml 0157 --dst_root stage2/moyashii/side/v11/0157 --options folds=[1,2]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0159.yaml 0159 --dst_root stage2/moyashii/side/v11/0159 --options folds=[0]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0177.yaml 0177 --dst_root stage2/moyashii/side/v11/0177 --options folds=[2,3]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0191.yaml 0191 --dst_root stage2/moyashii/side/v11/0191 --options folds=[3,4]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0200.yaml 0200 --dst_root stage2/moyashii/side/v11/0200 --options folds=[0,4]
   python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0212.yaml 0212 --dst_root stage2/moyashii/side/v11/0212 --options folds=[3,4]

   ### suguuuuu
   # Train the side classification models (using dataset v8 with pseudo labels v1)
   python3 src/stage2/suguuuuu/train.py src/stage2/suguuuuu/config/2064.yaml 2064_base-conv-batch2 --dst_root stage2/suguuuuu/side/v8/2064 --options folds=[2,3]
   ```

4. Prediction

   For prediction, first use the keypoint detection models to estimate the keypoints on the test images. Then, crop the images based on these keypoints and use the classification models to predict the class:

   ```bash
   python3 src/predict.py
   ```

### Notes

- **Disk Space**: Model training and prediction require a large amount of disk space. Ensure there is sufficient free space.


## Model Weights Correspondence Table

Pretrained model weights are not included in this repository due to their large file size. The weights will be automatically downloaded during the data preparation step, which is handled by the `src/utils/prepare_data.py` script.

The pretrained weights are hosted on a Kaggle Dataset: [RSNA 2024 LSDC 3rd Place Models](https://www.kaggle.com/datasets/akinosora/rsna2024-lsdc-3rd-models-pub).

The table below shows the correspondence between the pretrained model weights used in the Kaggle submission and the weights generated by the reproducible training code in this repository.

| No. | Stage | Part Name | Description                | Kaggle Submission Weight Path                          | Reproducibility Code Weight Path                       |
| --- | ----- | --------- | -------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| 1   | 1     | Moyashii  | Axial Keypoint Detector    | weight/stage1/moyashii/v5/0003/*best_score.pth         | output/stage1/moyashii/v5/0003/*best_score.pth         |
| 2   | 1     | Moyashii  | Sagittal Keypoint Detector | weight/stage1/moyashii/v6/0002/*best_score.pth         | output/stage1/moyashii/v6/0002/*best_score.pth         |
| 3   | 1     | Moyashii  | Sagittal Keypoint Detector | weight/stage1/moyashii/v6/0003/*best_score.pth         | output/stage1/moyashii/v6/0003/*best_score.pth         |
| 4   | 1     | tkmn      | Axial Keypoint Detector    | weight/stage1/tkmn/v5/0003/*best_score.pth             | output/stage1/tkmn/v5/0003/*best_score.pth             |
| 5   | 1     | tkmn      | Sagittal Keypoint Detector | weight/stage1/tkmn/v6/0002/*best_score.pth             | output/stage1/tkmn/v6/0002/*best_score.pth             |
| 6   | 1     | tkmn      | Sagittal Keypoint Detector | weight/stage1/tkmn/v6/0003/*best_score.pth             | output/stage1/tkmn/v6/0003/*best_score.pth             |
| 7   | 2     | Moyashii  | Center Classifier          | weight/stage2/moyashii/center/v11/0113/*best_score.pth | output/stage2/moyashii/center/v11/0113/*best_score.pth |
| 8   | 2     | Moyashii  | Center Classifier          | weight/stage2/moyashii/center/v11/0152/*best_score.pth | output/stage2/moyashii/center/v11/0152/*best_score.pth |
| 9   | 2     | Moyashii  | Center Classifier          | weight/stage2/moyashii/center/v11/0176/*best_score.pth | output/stage2/moyashii/center/v11/0176/*best_score.pth |
| 10  | 2     | Moyashii  | Center Classifier          | weight/stage2/moyashii/center/v11/0188/*best_score.pth | output/stage2/moyashii/center/v11/0188/*best_score.pth |
| 11  | 2     | Moyashii  | Center Classifier          | weight/stage2/moyashii/center/v11/0196/*best_score.pth | output/stage2/moyashii/center/v11/0196/*best_score.pth |
| 12  | 2     | Moyashii  | Center Classifier          | weight/stage2/moyashii/center/v11/0211/*best_score.pth | output/stage2/moyashii/center/v11/0211/*best_score.pth |
| 13  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0123/*best_score.pth      | output/stage2/tkmn/center/v9/0123/*best_score.pth      |
| 14  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0141/*best_score.pth      | output/stage2/tkmn/center/v9/0141/*best_score.pth      |
| 15  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0143/*best_score.pth      | output/stage2/tkmn/center/v9/0143/*best_score.pth      |
| 16  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0162/*best_score.pth      | output/stage2/tkmn/center/v9/0162/*best_score.pth      |
| 17  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0188/*best_score.pth      | output/stage2/tkmn/center/v9/0188/*best_score.pth      |
| 18  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0189/*best_score.pth      | output/stage2/tkmn/center/v9/0189/*best_score.pth      |
| 19  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0202/*best_score.pth      | output/stage2/tkmn/center/v9/0202/*best_score.pth      |
| 20  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0204/*best_score.pth      | output/stage2/tkmn/center/v9/0204/*best_score.pth      |
| 21  | 2     | tkmn      | Center Classifier          | weight/stage2/tkmn/center/v9/0211/*best_score.pth      | output/stage2/tkmn/center/v9/0211/*best_score.pth      |
| 22  | 2     | Moyashii  | Side Classifier            | weight/stage2/moyashii/side/v11/0082/*best_score.pth   | output/stage2/moyashii/side/v11/0082/*best_score.pth   |
| 23  | 2     | Moyashii  | Side Classifier            | weight/stage2/moyashii/side/v11/0157/*best_score.pth   | output/stage2/moyashii/side/v11/0157/*best_score.pth   |
| 24  | 2     | Moyashii  | Side Classifier            | weight/stage2/moyashii/side/v11/0159/*best_score.pth   | output/stage2/moyashii/side/v11/0159/*best_score.pth   |
| 25  | 2     | Moyashii  | Side Classifier            | weight/stage2/moyashii/side/v11/0177/*best_score.pth   | output/stage2/moyashii/side/v11/0177/*best_score.pth   |
| 26  | 2     | Moyashii  | Side Classifier            | weight/stage2/moyashii/side/v11/0191/*best_score.pth   | output/stage2/moyashii/side/v11/0191/*best_score.pth   |
| 27  | 2     | Moyashii  | Side Classifier            | weight/stage2/moyashii/side/v11/0200/*best_score.pth   | output/stage2/moyashii/side/v11/0200/*best_score.pth   |
| 28  | 2     | Moyashii  | Side Classifier            | weight/stage2/moyashii/side/v11/0212/*best_score.pth   | output/stage2/moyashii/side/v11/0212/*best_score.pth   |
| 29  | 2     | suguuuuu  | Side Classifier            | weight/stage2/suguuuuu/side/v8/2064/*best_score.pth    | output/stage2/suguuuuu/side/v8/2064/*best_score.pth    |
| 30  | 2     | tkmn      | Side Classifier            | weight/stage2/tkmn/side/v9/0101/*best_score.pth        | output/stage2/tkmn/side/v9/0101/*best_score.pth        |
| 31  | 2     | tkmn      | Side Classifier            | weight/stage2/tkmn/side/v9/0123/*best_score.pth        | output/stage2/tkmn/side/v9/0123/*best_score.pth        |
| 32  | 2     | tkmn      | Side Classifier            | weight/stage2/tkmn/side/v9/0124/*best_score.pth        | output/stage2/tkmn/side/v9/0124/*best_score.pth        |
| 33  | 2     | tkmn      | Side Classifier            | weight/stage2/tkmn/side/v9/0138/*best_score.pth        | output/stage2/tkmn/side/v9/0138/*best_score.pth        |
| 34  | 2     | tkmn      | Side Classifier            | weight/stage2/tkmn/side/v9/0139/*best_score.pth        | output/stage2/tkmn/side/v9/0139/*best_score.pth        |
| 35  | 2     | tkmn      | Side Classifier            | weight/stage2/tkmn/side/v9/0145/*best_score.pth        | output/stage2/tkmn/side/v9/0145/*best_score.pth        |
| 36  | 2     | tkmn      | Side Classifier            | weight/stage2/tkmn/side/v9/0203/*best_score.pth        | output/stage2/tkmn/side/v9/0203/*best_score.pth        |

## License

This project is licensed under the MIT License.

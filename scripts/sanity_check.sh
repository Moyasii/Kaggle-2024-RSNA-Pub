#!/bin/bash
set -e

kaggle datasets download akinosora/rsna2024-kpt-dataset-v6
unzip rsna2024-kpt-dataset-v6.zip
rm rsna2024-kpt-dataset-v6.zip

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@        Download and Preprocess the Data          @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
python3 src/utils/prepare_data.py
python3 src/utils/generate_kfold.py

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@        Train Keypoint Detection Models           @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
### Moyashii
# Create sagittal dataset v2 and train the sagittal keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v2.py moyashii 2024
python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v2_0008_config.yaml 0008 --dst_root stage1/moyashii/v2/0008 --options epochs=2
# Create sagittal dataset v4 and train the sagittal keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v4.py moyashii 2024
python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v4_0005_config.yaml 0005 --dst_root stage1/moyashii/v4/0005 --options epochs=2
python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v4_0012_config.yaml 0012 --dst_root stage1/moyashii/v4/0012 --options epochs=2
# Create sagittal dataset v6 and train the sagittal keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v6.py moyashii 2024
python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v6_0002_config.yaml 0002 --dst_root stage1/moyashii/v6/0002 --options epochs=2
python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v6_0003_config.yaml 0003 --dst_root stage1/moyashii/v6/0003 --options epochs=2
# Create axial dataset v5 and train the axial keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v5.py moyashii 2024
python3 src/stage1/moyashii/axial/train.py src/stage1/moyashii/axial/config/v5_0003_config.yaml 0003 --dst_root stage1/moyashii/v5/0003 --options epochs=2

python3 src/utils/check_files.py weight/stage1/moyashii output/stage1/moyashii
cp -r weight/stage1/moyashii output/stage1

### tkmn
# Create sagittal dataset v2 and train the sagittal keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v2.py tkmn 42
python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v2_0008_config.yaml 0008 --dst_root stage1/tkmn/v2/0008 --options epochs=2
# Create sagittal dataset v4 and train the sagittal keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v4.py tkmn 42
python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v4_0005_config.yaml 0005 --dst_root stage1/tkmn/v4/0005 --options epochs=2
python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v4_0012_config.yaml 0012 --dst_root stage1/tkmn/v4/0012 --options epochs=2
# Create sagittal dataset v6 and train the sagittal keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v6.py tkmn 42
python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v6_0002_config.yaml 0002 --dst_root stage1/tkmn/v6/0002 --options epochs=2
python3 src/stage1/moyashii/sagittal/train.py src/stage1/tkmn/sagittal/config/v6_0003_config.yaml 0003 --dst_root stage1/tkmn/v6/0003 --options epochs=2
# Create axial dataset v5 and train the axial keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v5.py tkmn 42
python3 src/stage1/moyashii/axial/train.py src/stage1/tkmn/axial/config/v5_0003_config.yaml 0003 --dst_root stage1/tkmn/v5/0003 --options epochs=2

python3 src/utils/check_files.py weight/stage1/tkmn output/stage1/tkmn
cp -r weight/stage1/tkmn output/stage1

cp train_5folds.csv input/processed/moyashii/keypoint_datasets/v6
cp train_5folds.csv input/processed/tkmn/keypoint_datasets/v6

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@          Train Classification Models             @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
### tkmn
# Create the dataset using keypoint detection models
python3 src/stage2/moyashii/tools/create_dastaset_v9.py tkmn
# Train the center classification models (using dataset v9)
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0022.yaml S0022 --dst_root stage2/tkmn/center/v9/0022 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0046.yaml S0046 --dst_root stage2/tkmn/center/v9/0046 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0051.yaml S0051 --dst_root stage2/tkmn/center/v9/0051 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0056.yaml S0056 --dst_root stage2/tkmn/center/v9/0056 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0078.yaml S0078 --dst_root stage2/tkmn/center/v9/0078 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0123.yaml S0123 --dst_root stage2/tkmn/center/v9/0123 --options folds=[0,1] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0141.yaml S0141 --dst_root stage2/tkmn/center/v9/0141 --options folds=[1,2] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0143.yaml S0143 --dst_root stage2/tkmn/center/v9/0143 --options folds=[2] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0162.yaml S0162 --dst_root stage2/tkmn/center/v9/0162 --options folds=[3] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0188.yaml S0188 --dst_root stage2/tkmn/center/v9/0188 --options folds=[0,1] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0189.yaml S0189 --dst_root stage2/tkmn/center/v9/0189 --options folds=[0,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0202.yaml S0202 --dst_root stage2/tkmn/center/v9/0202 --options folds=[2] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0204.yaml S0204 --dst_root stage2/tkmn/center/v9/0204 --options folds=[2,3] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/S0211.yaml S0211 --dst_root stage2/tkmn/center/v9/0211 --options folds=[3,4] debug.use_debug=true
# Train the side classification models (using dataset v9)
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0024.yaml 0024 --dst_root stage2/tkmn/side/v9/0024 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0057.yaml 0057 --dst_root stage2/tkmn/side/v9/0057 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0068.yaml 0068 --dst_root stage2/tkmn/side/v9/0068 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0093.yaml 0093 --dst_root stage2/tkmn/side/v9/0093 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0101.yaml 0101 --dst_root stage2/tkmn/side/v9/0101 --options folds=[0,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0123.yaml 0123 --dst_root stage2/tkmn/side/v9/0123 --options folds=[0,1] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0124.yaml 0124 --dst_root stage2/tkmn/side/v9/0124 --options folds=[2] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0138.yaml 0138 --dst_root stage2/tkmn/side/v9/0138 --options folds=[0,1] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0139.yaml 0139 --dst_root stage2/tkmn/side/v9/0139 --options folds=[3,4] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0145.yaml 0145 --dst_root stage2/tkmn/side/v9/0145 --options folds=[1,2] debug.use_debug=true
python3 src/stage2/tkmn/train.py src/stage2/tkmn/config/0203.yaml 0203 --dst_root stage2/tkmn/side/v9/0203 --options folds=[1] debug.use_debug=true

python3 src/utils/check_files.py weight/stage2/tkmn output/stage2/tkmn

### Moyashii
# Create the dataset using keypoint detection models
python3 src/stage2/moyashii/tools/create_dastaset_v8.py moyashii
python3 src/stage2/moyashii/tools/create_dastaset_v9.py moyashii
python3 src/stage2/moyashii/tools/create_dastaset_v11.py moyashii
# Train the center classification models (using dataset v9)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0053.yaml 0053 --dst_root stage2/moyashii/center/v9/0053_42 --options folds=[0,1,2,3,4] seed=42 debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0053.yaml 0053 --dst_root stage2/moyashii/center/v9/0053_2024 --options folds=[0,1,2,3,4] seed=2024 debug.use_debug=true
# Train the side classification models (using dataset v9)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0054.yaml 0054 --dst_root stage2/moyashii/side/v9/0054_42 --options folds=[0,1,2,3,4] seed=42 debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0054.yaml 0054 --dst_root stage2/moyashii/side/v9/0054_2024 --options folds=[0,1,2,3,4] seed=2024 debug.use_debug=true
# Train the center classification models (using dataset v11)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0057.yaml 0057 --dst_root stage2/moyashii/center/v11/0057_42 --options folds=[0,1,2,3,4] seed=42 debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0057.yaml 0057 --dst_root stage2/moyashii/center/v11/0057_2024 --options folds=[0,1,2,3,4] seed=2024 debug.use_debug=true
# Train the side classification models (using dataset v11)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0058.yaml 0058 --dst_root stage2/moyashii/side/v11/0058_42 --options folds=[0,1,2,3,4] seed=42 debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0058.yaml 0058 --dst_root stage2/moyashii/side/v11/0058_2024 --options folds=[0,1,2,3,4] seed=2024 debug.use_debug=true
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
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0109.yaml 0109 --dst_root stage2/moyashii/center/v11/0109 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0113.yaml 0113 --dst_root stage2/moyashii/center/v11/0113 --options folds=[0,1,2,3,4] debug.use_debug=true
# Train the side classification models (using dataset v11 with pseudo labels v1)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0074.yaml 0074 --dst_root stage2/moyashii/side/v11/0074 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0082.yaml 0082 --dst_root stage2/moyashii/side/v11/0082 --options folds=[0,1,2,3,4] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0114.yaml 0114 --dst_root stage2/moyashii/side/v11/0114 --options folds=[0,1,2,3,4] debug.use_debug=true
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
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0152.yaml 0152 --dst_root stage2/moyashii/center/v11/0152 --options folds=[1] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0176.yaml 0176 --dst_root stage2/moyashii/center/v11/0176 --options folds=[3,4] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0188.yaml 0188 --dst_root stage2/moyashii/center/v11/0188 --options folds=[0,4] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0196.yaml 0196 --dst_root stage2/moyashii/center/v11/0196 --options folds=[2,3] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0211.yaml 0211 --dst_root stage2/moyashii/center/v11/0211 --options folds=[0,4] debug.use_debug=true
# Train the side classification models (using dataset v11 with pseudo labels v3)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0157.yaml 0157 --dst_root stage2/moyashii/side/v11/0157 --options folds=[1,2] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0159.yaml 0159 --dst_root stage2/moyashii/side/v11/0159 --options folds=[0] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0177.yaml 0177 --dst_root stage2/moyashii/side/v11/0177 --options folds=[2,3] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0191.yaml 0191 --dst_root stage2/moyashii/side/v11/0191 --options folds=[3,4] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0200.yaml 0200 --dst_root stage2/moyashii/side/v11/0200 --options folds=[0,4] debug.use_debug=true
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0212.yaml 0212 --dst_root stage2/moyashii/side/v11/0212 --options folds=[3,4] debug.use_debug=true

python3 src/utils/check_files.py weight/stage2/moyashii output/stage2/moyashii

### suguuuuu
# Train the side classification models (using dataset v8 with pseudo labels v1)
python3 src/stage2/suguuuuu/train.py src/stage2/suguuuuu/config/2064.yaml 2064_base-conv-batch2 --dst_root stage2/suguuuuu/side/v8/2064 --options folds=[2,3] debug.use_debug=true

python3 src/utils/check_files.py weight/stage2/suguuuuu output/stage2/suguuuuu

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@                     Predict                      @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
python3 src/predict.py

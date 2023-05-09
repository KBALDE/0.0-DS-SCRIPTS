#!bin/bash


DATA_FILE_PATH="./0.0-DS-SCRIPTS/TAB_DATA/application_train.csv"

MODEL_FILE="./0.0-DS-SCRIPTS/TAB_DATA/xgb_model.pkl"
TEST_FILE="./0.0-DS-SCRIPTS/TAB_DATA/xgb_test_dict.npy"

SCRIPT_ROOT="./0.0-DS-SCRIPTS/TAB_XGBOOST"

python $SCRIPT_ROOT/XGBoost.py   --target_type="BIN" \
                                 --target_col_name="TARGET" \
                                 --file_path=$DATA_FILE_PATH  \
                                 --model_filename=$MODEL_FILE \
                                 --test_filename=$TEST_FILE
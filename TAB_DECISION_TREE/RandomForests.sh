#!/bin/bash

DATA_FILE="./0.0-DS-SCRIPTS/TAB_DATA/application_train.csv"
MODEL_FILE="./0.0-DS-SCRIPTS/TAB_DATA/rf_model.pkl"
TEST_FILE="./0.0-DS-SCRIPTS/TAB_DATA/rf_test_dict.npy"
SCRIPT_ROOT="./0.0-DS-SCRIPTS/TAB_DECISION_TREE"

python $SCRIPT_ROOT/RandomForests.py  --target_type="BIN" \
                                      --target_col_name="TARGET" \
                                      --file_path=$DATA_FILE \
                                      --model_filename=$MODEL_FILE \
                                      --test_filename=$TEST_FILE  

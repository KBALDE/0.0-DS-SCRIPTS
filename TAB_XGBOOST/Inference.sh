#!/bin/bash

MODEL_FILE="./0.0-DS-SCRIPTS/TAB_DATA/xgb_model.pkl"
TEST_FILE="./0.0-DS-SCRIPTS/TAB_DATA/xgb_test_dict.npy"

SCRIPT_ROOT="./0.0-DS-SCRIPTS/TAB_XGBOOST"

python $SCRIPT_ROOT/Inference.py --test_file=$TEST_FILE --model_filename=$MODEL_FILE
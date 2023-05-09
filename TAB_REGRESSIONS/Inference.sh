#!/bin/bash

SCRIPT_ROOT="./0.0-DS-SCRIPTS/TAB_REGRESSIONS"
MODEL_FILE="./0.0-DS-SCRIPTS/TAB_DATA/reg_model.pkl"
TEST_FILE="./0.0-DS-SCRIPTS/TAB_DATA/reg_test_dict.npy"

python $SCRIPT_ROOT/Inference.py --test_file=$TEST_FILE  --model_filename=$MODEL_FILE
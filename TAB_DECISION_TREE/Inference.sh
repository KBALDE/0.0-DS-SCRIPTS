#!/bin/bash

MODEL_FILE="./0.0-DS-SCRIPTS/TAB_DATA/rf_model.pkl"
TEST_FILE="./0.0-DS-SCRIPTS/TAB_DATA/rf_test_dict.npy"
SCRIPT_ROOT="./0.0-DS-SCRIPTS/TAB_DECISION_TREE"


python $SCRIPT_ROOT/Inference.py --test_file=$TEST_FILE --model_filename=$MODEL_FILE

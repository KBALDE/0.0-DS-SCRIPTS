#!/bin/bash

MODEL_FILE="./0.0-DS-SCRIPTS/TAB_DATA/nn_model/cp.ckpt"
TEST_FILE="./0.0-DS-SCRIPTS/TAB_DATA/nn_test_dict.npy"

SCRIPT_ROOT="./0.0-DS-SCRIPTS/TAB_NEURAL_NET"

python $SCRIPT_ROOT/Inference.py   --target_type="BIN" --target_col_name="TARGET" \
                       --test_file=$TEST_FILE \
                       --input_dim=29 \
                       --input_layer_units=40 \
                       --num_hidden_layers=3  --hidden_layer_units=120 --activation='relu' \
                       --dropout=True \
                       --learning_rate=0.001 --checkpoint_path=$MODEL_FILE

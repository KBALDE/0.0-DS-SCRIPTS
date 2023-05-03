#!/bin/bash

python Inference.py   --target_type="BIN" --target_col_name="TARGET" \
                       --model_filename="model.pkl" --test_file="test_dict.npy" \
                       --input_dim=29 \
                       --input_layer_units=40 \
                       --num_hidden_layers=3  --hidden_layer_units=120 --activation='relu' \
                       --dropout=True \
                       --learning_rate=0.001 --checkpoint_path="model/cp.ckpt"

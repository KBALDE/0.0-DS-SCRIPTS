#!/bin/bash

python NeuralNetwork.py     --target_type="BIN" --num_epochs=2 --fine_tune=False \
                             --target_col_name="TARGET" \
                             --file_path="../TAB_DATA/application_train.csv" \
                             --model_filename="model.pkl"\
                             --input_dim=29 \
                             --input_layer_units=40 \
                             --num_hidden_layers=3  --hidden_layer_units=120 --activation='relu' \
                             --dropout=True \
                             --learning_rate=0.001 --checkpoint_path="model/cp.ckpt"

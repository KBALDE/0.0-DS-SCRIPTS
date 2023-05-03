#!/bin/bash

python RandomForests.py  --target_type="BIN" \
                          --target_col_name="TARGET" \
                          --file_path="./data/application_train.csv" \
                          --model_filename="model.pkl"

#!/bin/bash

python RandomForests.py  --target_type="BIN" \
                          --target_col_name="TARGET" \
                          --file_path="../TAB_DATA/application_train.csv" \
                          --model_filename="model.pkl"

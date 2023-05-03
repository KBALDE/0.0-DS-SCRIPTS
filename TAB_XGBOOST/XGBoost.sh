#!bin/bash

python XGBoost.py   --target_type="BIN" \
                    --target_col_name="TARGET" \
                    --file_path="../TAB_DATA/application_train.csv" \
                    --model_filename="model.pkl"

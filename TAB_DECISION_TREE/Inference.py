import joblib
import numpy as np


import argparse


# ARGS
parser = argparse.ArgumentParser()

parser.add_argument("--test_file", type=str, help="test_file")
parser.add_argument("--model_filename", type=str, help="model_filename")



args = parser.parse_args()

test_file=args.test_file
model_filename=args.model_filename



def main():
    # load model
    model=joblib.load(model_filename)
    
    # load test set
    # next try to go from raw data
    # test_file="test_dict.npy"
    test_dict=np.load(test_file, allow_pickle=True)
    test_dict=test_dict.tolist()
    
    # score
    print("Test Set Score : ", model.score(test_dict['X_test'], test_dict['y_test']))

if __name__=="__main__":
    main()

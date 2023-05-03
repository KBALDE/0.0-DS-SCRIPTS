import pandas as pd
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("--target_type", type=str, help="target_type")
parser.add_argument("--target_col_name", type=str, help="target_col_name")
parser.add_argument("--file_path", type=str, help="file_path")
parser.add_argument("--model_filename", type=str, help="model_filename")
parser.add_argument("--input_dim", type=int, help="input_dim")
parser.add_argument("--input_layer_units", type=int, help="input_layer_units")
parser.add_argument("--num_hidden_layers", type=int, help="num_hidden_layers")
parser.add_argument("--hidden_layer_units", type=int, help="hidden_layer_units")
parser.add_argument("--activation", type=str, help="activation")
parser.add_argument("--dropout", type=bool, help="dropout")
parser.add_argument("--learning_rate", type=float, help="learning_rate")
parser.add_argument("--checkpoint_path", type=str, help="checkpoint_path")
parser.add_argument("--num_epochs", type=int, help="num_epochs")
parser.add_argument("--test_file", type=str, help="test split saved file")


args = parser.parse_args()

target_type=args.target_type
target_col_name=args.target_col_name
input_dim=args.input_dim
input_layer_units=args.input_layer_units
num_hidden_layers=args.num_hidden_layers
hidden_layer_units=args.hidden_layer_units
activation=args.activation
dropout=args.dropout
learning_rate=args.learning_rate
file_path=args.file_path
model_filename=args.model_filename
checkpoint_path=args.checkpoint_path
num_epochs=args.num_epochs
test_file= args.test_file



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


class NeuralNetwork:
    def __init__(self, 
                 input_dim,
                 input_layer_units,
                 num_hidden_layers,
                 hidden_layer_units, 
                 activation, 
                 dropout,
                 learning_rate,
                 target_type='BIN'
                ):
        self.input_dim = input_dim
        self.input_layer_units = input_layer_units
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_units = hidden_layer_units 
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        if target_type=='BIN':
            self.output_activation='sigmoid'
            self.output_units=1
            self.loss_function=tf.losses.BinaryCrossentropy()
            self.metrics=tf.keras.metrics.BinaryAccuracy()
        elif target_type=='MUL':
            self.output_activation = 'softmax'
            self.output_units=3 ## to be processed, to be taking from the data
            self.loss_function=tf.losses.CategoricalCrossentropy()
            self.metrics=tf.keras.losses.CategoricalCrossentropy()
        elif target_type=='QNT':
            self.output_activation='linear'
            self.output_units=1
            self.loss_function=tf.keras.losses.mean_squared_error
            self.metrics=tf.keras.losses.mean_squared_error
        else:
            print("You need to provide the target_type as BIN, MUL or QNT")

       

        self.model = keras.Sequential()
        
        self.model.add(keras.layers.Input(shape=(self.input_dim,)))
        
        self.model.add(keras.layers.Dense(units=self.input_layer_units, 
                                          activation=self.activation,
                                          #kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                         )
                      )
        for _ in range(self.num_hidden_layers):
            self.model.add(keras.layers.Dense(units=self.hidden_layer_units, 
                                              activation=self.activation,
                                              #kernel_regularizer=tf.keras.regularizers.l2(0.1)
                                             ))
        if self.dropout:
            self.model.add(keras.layers.Dropout(rate=0.25))
        self.model.add(keras.layers.Dense(units=self.output_units,
                                          activation=self.output_activation))
        self.model.compile(
            loss=self.loss_function,
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[self.metrics]
        )
            
        
        
    
    def train(self, train_dict, val_dict, num_epochs, checkpoint_path):
        
        X_train=train_dict['X_train'].values
        y_train=train_dict['y_train'].values
        X_val=val_dict['X_val'].values
        y_val=val_dict['y_val'].values
        
        
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
            
        self.model.fit(X_train, y_train,
                       validation_data = (X_val, y_val), 
                       epochs = num_epochs,
                       callbacks=[cp_callback]
                      )

       
    def infer(self, test_dict):
        return self.model.predict(test_dict['X_test'].head(1))
    
def main():

    nn=NeuralNetwork(target_type=target_type,
                     input_dim=input_dim,
                     input_layer_units=input_layer_units,
                     num_hidden_layers=num_hidden_layers,
                     hidden_layer_units=hidden_layer_units, 
                     activation=activation, 
                     dropout=dropout,
                     learning_rate=learning_rate
                    )
        

                    
    
    #print(nn.summary())
    
    test_dict=np.load(test_file, allow_pickle=True).tolist()
    print(test_dict.keys())

    
    nn.model.load_weights(checkpoint_path)
    
    #print(model.model.summary())
    
    # score
    print("Test Set Score : ", nn.model.evaluate(test_dict['X_test'], test_dict['y_test']))
    
    
    
    
    
if __name__=="__main__":
    main()

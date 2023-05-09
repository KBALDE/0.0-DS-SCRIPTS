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
parser.add_argument("--fine_tune", type=bool, help="fine_tune")
parser.add_argument("--test_filename", type=str, help="test_filename")

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
fine_tune=args.fine_tune
test_filename=args.test_filename


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import keras_tuner

def set_index(df):
    
    index = [c for c in df.columns.tolist() if len(df[c].unique())==df.shape[0] ]
    
    if index:
        df.set_index(index[0], inplace=True)
    else:
        df['INDEX']=range(0, df.shape[0])
        df.set_index('INDEX', inplace=True)



# Function to calculate missing values by column
def delete_missing_columns(df, pct_missing= 30):
    """Returns df with colimns where missing values are than 30%"""
    
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table = mis_val_table.rename(columns = {0 : 'NumberMissingValues',
                                                                1 : 'PerctgMissingValues'})
    mis_val_table = mis_val_table.sort_values('PerctgMissingValues', 
                                                                    ascending=False).round(1)
    
    keep_columns = mis_val_table.loc[mis_val_table.PerctgMissingValues <= pct_missing].index.tolist()
    
        
    return df[keep_columns] 

def scale_std(df, col_name):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    df[col_name] = scaler.fit_transform(df[[col_name]])
    return df

def get_cat_nocat_df(df):
    
    cat=df.select_dtypes("object").columns.to_list()
    num_int= df.select_dtypes(["int"]).columns.to_list()
    num_float=df.select_dtypes(["float"]).columns.tolist()
    
    for c in df[num_int].columns:
        if len(df[c].unique()) < 100 :
            cat.append(c)
        else:
            num_float.append(c)
    
    return df[cat], df[num_float]
    

from sklearn.preprocessing import StandardScaler
def standardize_df(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df),
                        columns=df.columns, index= df.index)  

from sklearn.preprocessing import StandardScaler

def fillna_standardize_df(df):
    
    df=df.fillna(df.median())
    
    scaler = StandardScaler()
    
    return pd.DataFrame(scaler.fit_transform(df),
                        columns=df.columns, index= df.index)


def merge_cat_nocat(df_cat, df_nocat, target):
    return df_cat.merge(df_nocat, left_index=True,
                        right_index=True).merge(target, left_index=True, right_index=True)

def merge_df(df, target):
    return df.merge(target, left_index=True, right_index=True)

def process_input_data(df:pd.DataFrame(), target_col_name:str):

    set_index(df)

    
    target=df.pop(target_col_name)
    

    df = delete_missing_columns(df, pct_missing= 30)

    df_cat, df_val = get_cat_nocat_df(df)
    
    if df_val.empty == False:
        df_val = fillna_standardize_df(df_val)
        df = merge_df(df_val, target)
    
    if df_cat.empty == False :
        df_cat = balance_cat_data(df_cat)
        df_cat = pd.get_dummies(df_cat, drop_first=True)
        if df_val.empty == False:
            df = merge_df(df, df_cat)
        else:
            df = merge_df(target, df_cat)
            
    if df[target_col_name].dtype == 'float':
        df = scale_std(df, target_col_name)
        
    return df


def balance_cat_data(df_cat):
    for c in df_cat.columns.tolist():
        
        if (df_cat[c].value_counts(normalize=True).values[0] > 0.9) :
            df_cat.drop(columns=[c], inplace=True)
    
        elif (df_cat[c].value_counts(normalize=True).values[0] > 0.5) :
             df_cat.loc[:, str(c)]= np.where(df_cat[c]!= df_cat[c].value_counts(normalize=True).index[0],
                                        'X', df_cat[c])
        else:
            df_cat.loc[:, str(c)]= np.where(df_cat[c]!= df_cat[c].value_counts(normalize=True).index[0],
                                np.where(df_cat[c]!= df_cat[c].value_counts(normalize=True).index[1],
                                         'X',df_cat[c] ), df_cat[c])
    
    return df_cat



# une methode robuste pour indentifier les points aberrants
from sklearn.covariance import EllipticEnvelope

# identify outliers in a dataset
def outLiersMultiEnveloppe(df, listColBracket):
    '''It returns df without outliers'''
    
    ee = EllipticEnvelope(contamination=0.01)
    
    X = df[listColBracket]
    yhat = ee.fit_predict(X)
    
    mask = yhat != -1
    mask_out = yhat != 1
    
    out = X.loc[mask_out, :].index
    df = df[~df.index.isin(out)]
    
    return df

def build_pca_data(df, scale=False):
    ''' take a quantitative dataframe
        return pca.components_, pca instance, and df of coord. ind and PCAs
    '''
    # PCA libraries
    from sklearn.decomposition import PCA
    
    n_components=df.shape[1]
    pca_columns=['PC_'+ str(i) for i in range(1,n_components+1)]
    X = df.values
    
    if scale:  
        scaler=StandardScaler()
        X = scaler.fit_transform(X) 
    
    pca = PCA(n_components)
    df_v=pca.fit_transform(X)
    
    return  pd.DataFrame(pca.components_, index=pca_columns, columns=df.columns), \
                       pca, pd.DataFrame(df_v, index=df.index, columns=pca_columns)


# Choose MAIN_VAR, PCA_VAR
def main_var_pca_var(df, target_col_name):
    MAIN_VAR=[]
    for k in df.columns:
        if (abs(np.corrcoef(df[target_col_name],df[k])[0][1]) > 0.2):
            MAIN_VAR.append(k)
    PCA_VAR=[]
    for x in df.columns:
        if x in MAIN_VAR:
            continue
        else:
            PCA_VAR.append(x)
            
    return df[MAIN_VAR], df[PCA_VAR]


def balance_target(df, target_col_name):
    
    df= df.sort_values(by=df.columns.tolist(), ascending=False)
    
    balance_size = df[df[target_col_name]==1].shape[0]
    
    return df[df[target_col_name]==1].append(df[df[target_col_name]==0].iloc[:balance_size]).sample(frac=1)


def create_dataset(file_name, target_col_name, rebalance=False):
    
    # read
    df = pd.read_csv(file_name)
    
    # process
    df = process_input_data(df, target_col_name)

    # balance
    if rebalance:
        df = balance_target(df, target_col_name)
        
    if len(df.columns.tolist()) > 20 :
        # main var and pca var
        df_var, df_pca=main_var_pca_var(df,target_col_name)
        # process pca varz
        mat, comp, pca_table= build_pca_data(df_pca)
        
        df_clean= df_var.merge(pca_table[['PC_'+str(i) for i in range(1, int(df_pca.shape[1]/2+5))]]
                                  , left_index=True, right_index=True)
    else:
        df_clean = df
        
    # delete y_clean
    y_clean = df_clean[[target_col_name]]
    
    df_clean.drop(columns=[target_col_name], inplace=True)

    # merge
    return y_clean, df_clean


def split_data(X_df, y_df, target_col_name):
    
    from sklearn.model_selection import train_test_split
    
    if y_df[target_col_name].dtype=='int':
        strat_y=None
        strat_yt=None
        
    else :
        strat_y = y_df.values
  
        
       
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, 
                                                       test_size=0.33, 
                                                       random_state=42,
                                                       stratify=strat_y
                                                       )
    if strat_y:
        strat_yt=y_test.values
    
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, 
                                                    test_size=0.5, 
                                                    random_state=42,
                                                    stratify=strat_yt
                                                   )
    
    train_dict={'X_train': X_train, 'y_train': y_train}
    val_dict={'X_val': X_val, 'y_val': y_val}
    test_dict={'X_test': X_test, 'y_test': y_test}
        
    return train_dict, val_dict, test_dict

def prepare_data(file_path, 
                 target_col_name, 
                 rebalance=True):
    
    y_train, df_train = create_dataset(file_path, 
                                       target_col_name, 
                                       rebalance=True)
    
    train_dict, val_dict, test_dict = split_data(df_train, y_train, target_col_name)
    
    
    return train_dict, val_dict, test_dict

def nn_model(input_dim,
             num_hidden_layers,
             hidden_layer_units,
             activation,
             output_activation,
             output_units,
             dropout,
             learning_rate,
             metrics,
             loss_function
            ):
    
    model = keras.Sequential()
    
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    for _ in range(num_hidden_layers):
        model.add(keras.layers.Dense(units=hidden_layer_units,
                                     activation=activation)
                      )
    if dropout:
        model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(units=output_units,
                                 activation=output_activation))
    model.compile(loss=loss_function,
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[metrics])
    
    return model


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
    
    # data prepare
    y_train, df_train = create_dataset(file_path, 
                                       target_col_name=target_col_name, 
                                       rebalance=True)
    ## split
    train_dict, val_dict, test_dict= split_data(df_train, 
                                                y_train, 
                                                target_col_name)
    ## save the test_set for inference
    #np.save("test_dict.npy", test_dict)
    np.save(test_filename, test_dict)
    
    # instantiate the model
    nn=NeuralNetwork(target_type=target_type,
                     input_dim=input_dim,
                     input_layer_units=input_layer_units,
                     num_hidden_layers=num_hidden_layers,
                     hidden_layer_units=hidden_layer_units, 
                     activation=activation, 
                     dropout=dropout,
                     learning_rate=learning_rate
                    )
    

    nn.train(train_dict, val_dict, num_epochs, checkpoint_path)
    print("Training is done!")
    print("Test Set Score : ", nn.model.evaluate(test_dict['X_test'], test_dict['y_test']))
    
    
    
    
    
if __name__=="__main__":
    main()

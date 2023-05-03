
import pandas as pd
import numpy as np

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from xgboost import XGBClassifier, XGBRegressor

import argparse
import joblib


# ARGS
parser = argparse.ArgumentParser()

parser.add_argument("--target_col_name", type=str, help="target_col_name")
parser.add_argument("--file_path", type=str, help="file_path")
parser.add_argument("--model_filename", type=str, help="model_filename")
parser.add_argument("--target_type", type=str, help="target_type")


args = parser.parse_args()

target_col_name=args.target_col_name
file_path=args.file_path
model_filename=args.model_filename
target_type=args.target_type




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


class XGBoost:
    def __init__(self, target_type='BIN'):
        
        self.space = {'gamma': Real(0.001,1), 
                      'learning_rate': Real(0.001,1), 
                      'colsample_bylevel': Real(0.001,1), 
                      'colsample_bytree': Real(0.001,0.6), 
                      'colsample_bynode': Real(0.001, 1), 
                      'max_depth': Integer(2,8), 
                      'n_estimators': Integer(500,12000) }
        
        if target_type=='BIN':
            # cond for non balanced 
            self.model =  XGBClassifier(booster='gbtree', objective='binary:logistic', reg_lambda=0.1,
                                        learning_rate=0.1, n_estimators=10000, random_state=2, n_jobs=-1,
                                        #scale_pos_weight=None,
                                        #tree_method='gpu_hist',
                                        use_label_encoder=False)
            
            
        elif target_type=='MUL':
            self.model = XGBClassifier(booster='gbtree', objective='multi:softprob', reg_lambda=0.1,
                                       learning_rate=0.1, n_estimators=10000, random_state=2, n_jobs=-1,
                                       #tree_method='gpu_hist',
                                       use_label_encoder=False,
                                       )    
            

        elif target_type=='QNT':
            self.model = XGBRegressor(booster='gbtree', objective='reg:squarederror', reg_lambda=0.1,
                                      learning_rate=0.1, n_estimators=10000, random_state=2, n_jobs=-1,
                                      #tree_method='gpu_hist',
                                      use_label_encoder=False,
                                      )

        else:
            print("Please provide the your target variable type as BIN, MUL or QNT")
    
    
    def train(self, train_dict, val_dict, fine_tune=False):
        
        if fine_tune==False:
            self.model.fit(train_dict['X_train'], train_dict['y_train'])
            print("Training Score: ", self.model.score(train_dict['X_train'], train_dict['y_train']))
            print("Validation Score: ", self.model.score(val_dict['X_val'], val_dict['y_val']))
        else:
            self.model = BayesSearchCV(estimator = self.model,
                                        search_spaces = self.space, 
                                        cv = 3,
                                        n_iter = 30,   
                                        verbose = 0,
                                        refit = True,
                                        scoring='accuracy',
                                        random_state = 42
                                      )
            self.model.fit(train_dict['X_train'], train_dict['y_train'])
            print("Training Score Optimized: ", self.model.score(train_dict['X_train'], train_dict['y_train']))
            print("Validation Score Optimized: ", self.model.score(val_dict['X_val'], val_dict['y_val']))

    def infer(self, test_dict):
        return self.model.predict(test_dict['X_test'].head(1))

    
def main():

    xgb=XGBoost(target_type=target_type)
    
    
    
    # data prepare
    y_train, df_train = create_dataset(file_path, 
                                           target_col_name=target_col_name, 
                                           rebalance=True)
    # split
    train_dict, val_dict, test_dict= split_data(df_train, 
                                                    y_train, 
                                                    target_col_name)
    # save the test_set for inference
    np.save("test_dict.npy", test_dict)
    
    
    # train
    # include cond for FT
    xgb.train(train_dict, val_dict, fine_tune=False)
    
    # save the model
    _=joblib.dump(xgb.model, model_filename, compress=True)
    
    
if __name__=="__main__":
    main()

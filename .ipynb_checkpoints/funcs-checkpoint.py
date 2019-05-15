import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
    

import warnings
warnings.filterwarnings("ignore")

def get_dist_graphs(df):
    
    # We will do mean inputation for NaNs since we cannot plot with them.
    df = df.fillna(df.mean())

     # First we need to get the continuos variables.
     # Piazza question indicated that continuos variables are the ones that are not type object.

    k = 0
    continuos = []
    for t in df.dtypes:
        if t != 'object':
            continuos.append(df.columns.values[k])
        k += 1

    # Set up the matplotlib figure
    f, axes = plt.subplots(12,3, figsize=(120, 120))

    # Make font bigger.
    sns.set(font_scale = 2.5)

    count = 0

    for i in range(0,3):
        for j in range(0,12):
            sns.distplot(df[continuos[count]], ax = axes[j,i])
            axes[j,i].set_title(continuos[count],fontsize=50) #Added title 
            # Remove labels.
            axes[j,i].set_xlabel('');
            count += 1

    plt.show()
    
def get_scatter_graphs(df):
    
    # We will do mean inputation for NaNs since we cannot plot with them.
    df = df.fillna(df.mean())

     # First we need to get the continuos variables.
     # Piazza question indicated that continuos variables are the ones that are not type object.

    k = 0
    continuos = []
    for t in df.dtypes:
        if t != 'object':
            continuos.append(df.columns.values[k])
        k += 1

    # Set up the matplotlib figure
    f, axes = plt.subplots(12,3, figsize=(120, 120))

    # Make font bigger.
    sns.set(font_scale = 2.5)

    count = 0

    for i in range(0,3):
        for j in range(0,12):
            sns.scatterplot(x = df[continuos[count]], y=df['SalePrice'], ax = axes[j,i])
            axes[j,i].set_title(continuos[count],fontsize=50) #Added title 
            # Remove labels.
            axes[j,i].set_xlabel('');
            axes[j,i].set_ylabel(''); # we know that y axis is Sale Price, no need to put it 40 times.
            count += 1
    plt.show()
    
    

def top3_r2(X_train,y_train):
    

    k = 0
    categorical = []
    for t in X_train.dtypes:
        if t != 'object':
            categorical.append(X_train.columns.values[k])
        k += 1

    # The first two are identifiers, so it is not usefull to use them because the do not add value.
    count = 2 
    r2 = []

    print('Gettin R squared values... \n')
    for i in range(0,len(categorical)-2): # Avoid first two will make a gap, so end for before.

        x_t = pd.DataFrame(X_train[categorical[count]]) # Get the sincle categorical variable
        X_h = OneHotEncoder(categories = 'auto').fit(x_t) # One Hot Encode it
        X_h = X_h.transform(x_t).toarray() # Get it into an array form

        # Get the r2 with c-val and linear regression
        # Auxiliar to see if r2 is lower than 0, it should be 0 in this case.
        r2_aux = np.mean(cross_val_score(LinearRegression(),X_h,y_train, cv = 10))

        if r2_aux >= 0: r2.append(r2_aux)
        else: r2.append(0)
        # Nice print to see results.
        print(categorical[count] + ': \t' +str(r2[-1]))

        count += 1 # Next variable

    # This is just to make it pretty :)
    print('\nThe top 3 categorical variables are: \n')
    top3_v = np.asarray(r2).argsort()[-3:][::-1] 
    top3_n = top3_v + [2]*3
    
    # Save for next task
    tops = []
    
    for top in range(0,3):
        tops.append(categorical[top3_n[top]])
        print(categorical[top3_n[top]] + ' with an R squared of: ' + str(r2[top3_v[top]]) + '\n')
        
    return tops

def top3_graphs(df,top3):

    # Take it back to normal.
    sns.set(font_scale = 1)

    # Set up the matplotlib figure
    f, axes = plt.subplots(1,3, figsize=(30, 6))


    for i in range(0,3):
        sns.scatterplot(x = df[top3[i]], y=df['SalePrice'], ax = axes[i])
    
    plt.show()
    

def regressions(train):
    
    train = train.fillna(train.mean())

    # Lets get X and Y
    X_train = train.drop('SalePrice', axis =1)
    y_train = train.SalePrice

    categorical = X_train.dtypes == object

    # Lets do the column transformer with Standar Scaler and One-Hot Encoder.
    preprocess = make_column_transformer(
        (StandardScaler(), ~categorical),
        (OneHotEncoder(handle_unknown = 'ignore'), categorical))


    model_LR = make_pipeline(preprocess, LinearRegression())
    model_R = make_pipeline(preprocess, Ridge())
    model_L = make_pipeline(preprocess, Lasso())
    model_EN = make_pipeline(preprocess, ElasticNet())

    print('\nLinear Regression score is: ' , np.mean(cross_val_score(model_LR,X_train,y_train, cv = 10)),'\n')
    print('Ridge score is: ' , np.mean(cross_val_score(model_R,X_train,y_train, cv = 10)),'\n')
    print('Lasso score is: ' , np.mean(cross_val_score(model_L,X_train,y_train, cv = 10)),'\n')
    print('ElasticNet score is: ' , np.mean(cross_val_score(model_EN,X_train,y_train, cv = 10)),'\n')
    
    
def regression_w_GS(train):
    
    # Mean imputation
    train = train.fillna(train.mean())
    # Other label for categorical
    train = train.fillna('Other')
    
    # Lets get X and Y
    X_train = train.drop('SalePrice', axis =1)
    y_train = train.SalePrice

    categorical = X_train.dtypes == object

    # Lets do the column transformer with Standar Scaler and One-Hot Encoder.
    preprocess = make_column_transformer(
        (StandardScaler(), ~categorical),
        (OneHotEncoder(handle_unknown = 'ignore'), categorical))

    model_LR = make_pipeline(preprocess, LinearRegression())
    model_R = make_pipeline(preprocess, Ridge())
    model_L = make_pipeline(preprocess, Lasso())
    model_EN = make_pipeline(preprocess, ElasticNet())

    # Linear Regression.

    param_grid_LR = {'linearregression__fit_intercept': (True,False),
                  'linearregression__normalize': (True,False)}

    grid_LR = GridSearchCV(model_LR,param_grid_LR, cv=10)
    grid_LR.fit(X_train, y_train)

    print('\t\t\t Linear Regression: \n')
    print(grid_LR.best_params_, '\t score: ',grid_LR.score(X_train, y_train),'\n')

    # Ridge.

    param_grid_R = {'ridge__alpha':np.logspace(-3, 3, num=13)}

    grid_R = GridSearchCV(model_R,param_grid_R, cv=10)
    grid_R.fit(X_train, y_train)

    print('\t\t\t Ridge: \n')
    print(grid_R.best_params_, '\t score: ',grid_R.score(X_train, y_train),'\n')

    # Lasso.

    param_grid_L = {'lasso__alpha':np.logspace(-3, 3, num=13)}

    grid_L = GridSearchCV(model_L,param_grid_L, cv=10)
    grid_L.fit(X_train, y_train)

    print('\t\t\t Lasso: \n')
    print(grid_L.best_params_, '\t score: ',grid_L.score(X_train, y_train),'\n')

    # ElasticNet.

    param_grid_EN = {'elasticnet__alpha':np.logspace(-4, 0, 13)}

    grid_EN = GridSearchCV(model_EN,param_grid_EN, cv=10)
    grid_EN.fit(X_train, y_train)

    print('\t\t\t ElasticNet: \n')
    print(grid_EN.best_params_, '\t score: ',grid_EN.score(X_train, y_train),'\n')
    
    return grid_LR,grid_R,grid_L,grid_EN

    
def viz_top_features(X_train,grid,name,number):
    
    number = -number # Get the top number of features you wish.
    top_features = []
    arr = grid.best_estimator_.named_steps[name].coef_.argsort()[number:][::-1]
    for top in arr:
        top_features.append(pd.get_dummies(X_train).columns.values[top])

    sns.boxplot(top_features,arr)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    
    plt.show()
    
    

def graph_top_features(X_train,grid,name,number):
    number = -number # Get the top number of features you wish.
    top_features = []
    arr = grid.best_estimator_.named_steps[name].coef_.argsort()[number:][::-1]
    for top in arr:
        top_features.append(pd.get_dummies(X_train).columns.values[top])
    return top_features
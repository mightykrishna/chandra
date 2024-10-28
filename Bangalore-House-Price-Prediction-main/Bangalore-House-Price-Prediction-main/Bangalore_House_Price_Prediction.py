import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv("Bengaluru_House_Data.csv")
#print(df.head())
#print(df.shape)

#for column in df.columns:
#    print(df[column].value_counts)
df.drop(columns=['area_type', 'availability', 'society', 'balcony'], inplace=True)
#print(df.isnull().sum())
df['location'] = df['location'].fillna('Sarjapur Road')
#print(df['size'].value_counts)
df['size'] = df['size'].fillna('2 BHK')
df['bath'] = df['bath'].fillna(df['bath'].median())
df['bhk'] = df['size'].str.split().str.get(0).astype(int)
#print(df[df.bhk > 20])

#print(df['total_sqft'].unique())

def ConvertRange(x):
    temp = x.split('-')
    if len(temp) == 2:
        return (float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None
    
df['total_sqft'] = df['total_sqft'].apply(ConvertRange)
#print(df.head())

df['price_per_sqft'] = df['price']*100000 / df['total_sqft']
#print(df['price_per_sqft'])

df['location'] = df['location'].apply(lambda x: x.strip())
location_count = df['location'].value_counts()

location_count_less_10 = location_count[location_count<=10]

df['location'] = df['location'].apply(lambda x:"Other" if x in location_count_less_10 else x)
#print(df.describe())

df = df[((df['total_sqft']/df['bhk']) >= 300)]

#print(df['price_per_sqft'].describe())
def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index=True)
    return df_output
df = remove_outliers_sqft(df)

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df = bhk_outlier_remover(df)

df.drop(columns = ['size','price_per_sqft'],inplace=True)

df.to_csv("New_Bangalore_houses_prices.csv")

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

column_trans = make_column_transformer((OneHotEncoder(),['location']),remainder='passthrough')
scaler = StandardScaler(with_mean=False)
lr = LinearRegression()

pipe = make_pipeline(column_trans,scaler,lr)
pipe.fit(X_train,y_train)
y_pred_lr = pipe.predict(X_test)
#print(r2_score(y_test, y_pred_lr))
'''
lasso = Lasso()
pipe = make_pipeline(column_trans,scaler,lasso)
pipe.fit(X_train,y_train)
y_pred_lasso = pipe.predict(X_test)
#print(r2_score(y_test,y_pred_lasso))'''
'''
ridge = Ridge()
pipe = make_pipeline(column_trans,scaler,ridge)
pipe.fit(X_train,y_train)
y_pred_ridge = pipe.predict(X_test)
#print(r2_score(y_test,y_pred_ridge))'''

print("No Regularization: ",r2_score(y_test, y_pred_lr))
#print("Lasso: ",r2_score(y_test,y_pred_lasso))
#print("Ridge: ",r2_score(y_test,y_pred_ridge))

pickle.dump(pipe, open('LrModel.pkl','wb'))
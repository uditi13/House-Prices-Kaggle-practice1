# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:37:51 2020

@author: Uditi
"""

# importing values
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df_train = pd.read_csv('C:\\Users\\Uditi\\Desktop\\house-prices-advanced-regression-techniques\\train.csv')

df_train.isnull().sum().sort_values(ascending = False)
# PoolQC , MiscFeature, Alley, Fence has a lot of null values

# heatmap to show missing values
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False)
# PoolQC , MiscFeature, Alley, Fence has a lot of null values

df_train.shape
df_train.info()

df_train['MSZoning'].value_counts()

# FILLLING MISSING VALUES

# filling nan values of LotFrontage w mean
df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())

# We will drop the features which have too many missing values
# dropping Alley
df_train.drop(['Alley'], axis=1, inplace=True)

# Filling missing values of categorical w mode
df_train['BsmtCond']=df_train['BsmtCond'].fillna(df_train['BsmtCond'].mode()[0])

df_train['BsmtQual']=df_train['BsmtQual'].fillna(df_train['BsmtQual'].mode()[0])

df_train['FireplaceQu']=df_train['FireplaceQu'].fillna(df_train['FireplaceQu'].mode()[0])

df_train['GarageType']=df_train['GarageType'].fillna(df_train['GarageType'].mode()[0])

df_train['GarageFinish']=df_train['GarageFinish'].fillna(df_train['GarageFinish'].mode()[0])

df_train['GarageQual']=df_train['GarageQual'].fillna(df_train['GarageQual'].mode()[0])

df_train['GarageCond']=df_train['GarageCond'].fillna(df_train['GarageCond'].mode()[0])

#The features with more than 50% missing values are dropped

df_train.drop(['GarageYrBlt'], axis=1, inplace=True)
df_train.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df_train.shape

df_train.drop(['Id'],axis=1,inplace=True)

# replacing more categorical null values w mode
df_train['BsmtFinType2']=df_train['BsmtFinType2'].fillna(df_train['BsmtFinType2'].mode()[0])
df_train['BsmtExposure']=df_train['BsmtExposure'].fillna(df_train['BsmtExposure'].mode()[0])

df_train['MasVnrType']=df_train['MasVnrType'].fillna(df_train['MasVnrType'].mode()[0])
df_train['MasVnrArea']=df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mode()[0])

df_train.dropna(inplace=True)

# HANDLING CATEGORICAL FEATURES
# list of columns containing all categorical features
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

len(columns)

# CATEGORICAL TO ONE HOT ENCODING
# fn which will be handling all features and converting
# it to categorical and will finally concatenate dataframe
# w the categorical features
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final




# making copy of the original dataframe
main_df=df_train.copy()


# Combining test data
df_test=pd.read_csv('C:\\Users\\Uditi\\Desktop\\house-prices-advanced-regression-techniques\\formulatedtest1.csv')

df_test.shape


# concatenating test and train
final_df=pd.concat([df_train,df_test],axis=0) # acc to rows


final_df.shape

# calling the fn
final_df=category_onehot_multcols(columns)

# combined 
final_df.shape # 235 columns

# removing the duplicated columns
final_df =final_df.loc[:,~final_df.columns.duplicated()]

#after removing duplicates
final_df.shape # 175 columns






from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error



# Splitting the data to training and testing data set
Train=final_df.iloc[:1422,:]
Test=final_df.iloc[1422:,:]

# dropping SalePrice from test dataset
# as all the values are nan
Test.drop(['SalePrice'],axis=1,inplace=True)

Test.shape


# MODEL BUILDING 
X_train=Train.drop(['SalePrice'],axis=1)
y_train=Train['SalePrice']



#Plotting the variable SalePrice
prices = pd.DataFrame({"1. Before":y_train, "2. After":np.log(y_train)})
prices.hist()
#We see that the before part is skewed and after is bell shaped or normal

# TRANSFORMING SALEPRICE AS LOG VALUE
# Here we use the strategy of taking the natural log of the price column,
# This is done because the data is very skewed in its natural form.
# After taking the natural log, we can understand it better.
y_train=np.log(y_train) 

train1_x,test1_x,train1_y,test1_y=train_test_split(X_train,y_train,test_size=0.3,random_state=3)
print(train1_x.shape, test1_x.shape, train1_y.shape, test1_y.shape)

# BASELINE MODEL

base_pred=np.mean(test1_y)
print(base_pred)

base_pred=np.repeat(base_pred,len(test1_y))

base_root_mean_square=np.sqrt(mean_squared_error(test1_y,base_pred))
print(base_root_mean_square)

# LINEAR REGRESSION

#Setting intercept as true
linear2=LinearRegression(fit_intercept=True)

#MODEL
linear2.fit(train1_x,train1_y)

#Predicting model on test set
predictions2=linear2.predict(test1_x)

#Computing RMSE for prediction
rmse2=np.sqrt(mean_squared_error(test1_y,predictions2))
print(rmse2)
# Since value of rmse of predictions is much less than the rmse of base model,
# Our linear regression model is working very well.


# R squared value
r2_test2=linear2.score(test1_x,test1_y)
r2_train2=linear2.score(train1_x,train1_y)
print(r2_test2,r2_train2)
# If the r2 scores for both test and train are similar, model is working well.


#Regression diagnostics - Residual plot analysis

# Residuals are the differences between the predicted data and test data.
# Residual plot has Residuals on the y-axis and the predictions for text_x on x-axis
residuals2=test1_y-predictions2
sns.regplot(x=predictions2,y=residuals2,scatter=True,fit_reg=False)
residuals2.describe()
# Closer the mean is to zero for residuals, the better. 
# Hence the errors should be less.
# Closer the scatter plot is to zero for residuals, the better.


# RANDOM FOREST REGRESSOR MODEL

#Model parameters
rf2= RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,
                          min_samples_split=10,min_samples_leaf=4,random_state=1)


# Model
rf2.fit(train1_x,train1_y)

# Predicting model on test set
predictions_rf2=rf2.predict(test1_x)

# Computing RMSE for prediction

rmse_rf2=np.sqrt(mean_squared_error(test1_y,predictions_rf2))
print(rmse_rf2)
# rmse of random forest is lesser than that of linear reg model hence this is better.
# Since value of rmse of predictions is much less than the rmse of base model,
# Our  model is working very well.

# R squared value
r2_test_rf2=rf2.score(test1_x,test1_y)
r2_train_rf2=rf2.score(train1_x,train1_y)
print(r2_test_rf2,r2_train_rf2)
# If the r2 scores for both test and train are similar, model is working well.


# RMSE VALUES FOUND : 
print('The rmse value found by linear regression method: ', rmse2) #0.17826123177987618
print('The rmse value found by random forest method: ',rmse_rf2)   #0.1401308719899409
# the lesser the rmse, the better is our model

y_pred=rf2.predict(Test)


#final SalePrice Predictions :
# Using Ramdom Forest model
y_pred = np.exp(y_pred)

print(y_pred)

# Converting the result into df
pred=pd.DataFrame(y_pred)
pred.to_csv('C:\\Users\\Uditi\\Desktop\\house-prices-advanced-regression-techniques\\PredictedSalePrice.csv', index=False)














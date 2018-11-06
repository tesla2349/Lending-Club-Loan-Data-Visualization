import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
    
# Read the dataset
df = pd.read_csv('data/loan.csv')

# Correlation matrix
numeric = ['int_rate','annual_inc','loan_amnt', 'installment', 'dti']
df_num = df[numeric]
df_num.corr()

# Independent variables
df_X = df.drop('int_rate',axis=1)

# Set the interest rate as dependent variable 
df_Y = df['int_rate']

# Use label encoder to transfer each categorical feature into numbers
le = preprocessing.LabelEncoder()
categ = ['term','grade','addr_state','issue_d','home_ownership','verification_status']
df_categ = df_X[categ].apply(le.fit_transform)
df_X = df_X.drop(categ, axis=1)
df_X = df_X.join(df_categ)
X = df_X.values
y = df_Y.values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Parameters of the random forest's model
parameters = {'n_estimators': 1,
              'max_features': "auto",
              'criterion': 'mse',
              'random_state': 0,
              'n_jobs': -1
              }

clf = RandomForestRegressor(**parameters)

# Train, test model
def train_test_model(clf, X_train, Y_train, X_test, Y_test):
    # Fit a model by providing X and y from training set
    clf.fit(X_train, Y_train)

    # Make prediction on the training data
    y_train_pred = clf.predict(X_train)

    # Make predictions on test data
    y_test_pred = clf.predict(X_test)

    # print model results
    train_rmse = sqrt(mean_squared_error(Y_train,y_train_pred))
    test_rmse = sqrt(mean_squared_error(Y_test,y_test_pred))
    print('train rmse is: %.3f'%train_rmse)
    print('test rmse is: %.3f'%test_rmse)


train_test_model(clf, X_train, y_train, X_test, y_test)

# Feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
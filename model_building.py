import pandas as pd
import numpy as np
# Load data train
train = pd.read_csv('datasets/train.csv')

# Data Preprocessing
## Outliers
from scipy.stats.mstats import winsorize
for v in ['account_length', 'total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 
          'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_charge']:
    train['trans_'+v] = winsorize(train[v],limits=[.01,.99])

for l in ['number_vmail_messages', 'total_intl_calls', 'number_customer_service_calls']:
  train['trans_'+l] = np.log(train[l]+1)

# Label Encoding
for col in ['international_plan', 'voice_mail_plan', 'churn']:
  train[col] = train[col].apply(lambda x: 1 if x == 'yes' else 0)

for cat in ['state', 'area_code']:
    onehots = pd.get_dummies(train[cat], prefix=cat, drop_first = True)
    train = train.join(onehots)

# Drop Unnecessary Features
train.drop(['state', 'area_code'], axis = 1, inplace = True)
train.drop(columns = ['account_length', 'total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 
          'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_charge', 'number_vmail_messages', 'total_intl_calls', 'number_customer_service_calls',
          'trans_total_day_minutes', 'trans_total_eve_minutes', 'trans_total_night_minutes', 'trans_total_intl_minutes'], inplace = True)

# Scaling Data
from sklearn.preprocessing import StandardScaler
X = train.drop('churn', axis = 1)
y = train['churn']

# StandardScaler data
# Initialize StandardScaler into scaler
scaler = StandardScaler()

# Fit X and transform menjadi X_scaled
X_scaled = scaler.fit_transform(X)

# assign 21 into SEED for reproductivity
SEED = 21

# Perform PCA with the chosen number of components and project data onto components
from sklearn.decomposition import PCA
pca = PCA(n_components = 55, random_state = SEED)

# Fit and transform the scaled training features using pca
X_scaled_pca = pca.fit_transform(X_scaled)

# Balance Data
# import library
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = SEED)

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X_scaled_pca, y)

# Machine Learning Modelling
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=SEED)
rf.fit(x_smote, y_smote)

# Saving the scaler, pca, model
import pickle
pickle.dump(scaler, open('scaler_clf.pkl', 'wb'))
pickle.dump(pca, open('pca_clf.pkl', 'wb'))
pickle.dump(rf, open('model_clf.pkl', 'wb'))


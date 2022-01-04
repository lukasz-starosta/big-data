
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

# Generating 600 samples using make_blobs
dataset, clusters = make_blobs(n_samples = 600, centers = 8, cluster_std = 0.75, random_state = 0)

X = pd.read_csv('../data/NYPD_Complaint_Data_Historic.csv', na_values=np.nan)

# get only selected columns from the dataset
data = X[['OFNS_DESC', 'BORO_NM', 'LOC_OF_OCCUR_DESC', 'PREM_TYP_DESC', 'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX']]

data = data[data['SUSP_SEX'].isin(['M', 'F'])]

# replace UNKNOWN for np.nan
data.replace('UNKNOWN', np.nan, inplace=True)

# delete records with unknown/no value
data.dropna(subset=['OFNS_DESC', 'BORO_NM', 'LOC_OF_OCCUR_DESC','PREM_TYP_DESC', 'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX'], inplace=True)

data = data[data['SUSP_AGE_GROUP'].isin(['<18', '18-24', '25-44', '45-64', '65+'])]

data = data[data['VIC_AGE_GROUP'].isin(['<18', '18-24', '25-44', '45-64', '65+'])]

X = data

Y = data['SUSP_RACE']

le = labelencoder = LabelEncoder()

X = X.apply(le.fit_transform)

# Handling the missing values
X.fillna(method ='ffill', inplace = True)

# Scaling the data so that all the features become comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizing the data so that the data approximately 
# follows a Gaussian distribution
X_normalized = normalize(X_scaled)

# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

X_imputed_df = pd.DataFrame(X_normalized, columns = X.columns)

print(X_imputed_df)

# Creating the BIRCH clustering model
model = Birch(branching_factor = 50, n_clusters = None, threshold = 0.2)

# Reducing the dimensionality of the Data
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
principalDf = pd.DataFrame(data = X_principal, columns = ['principal component 1', 'principal component 2'])

# Fit the data (Training)
model.fit(X_principal)
  
# Predict the same data
pred = model.predict(X_principal)

# Creating a scatter plot
plt.scatter(X_principal[:, 0], X_principal[:, 1], c = pred)
plt.show()
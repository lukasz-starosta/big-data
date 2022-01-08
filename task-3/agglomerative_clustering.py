
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

le = labelencoder = LabelEncoder()

Z = X.head(500)
print(Z)

X = X.apply(le.fit_transform)

X = X.head(500)

print(X)

# Handling the missing values
X.fillna(method ='ffill', inplace = True)

# Step 3: Preprocessing the data

# Scaling the data so that all the features become comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
  
# Normalizing the data so that the data approximately 
# follows a Gaussian distribution
X_normalized = normalize(X_scaled)
  
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

# Step 4: Reducing the dimensionality of the Data
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

X_principal = X_principal.head(100)

plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))

# k = 2
ac2 = AgglomerativeClustering(n_clusters = 2)

# # Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac2.fit_predict(X_principal), cmap ='rainbow')
plt.show()

# k = 3
ac2 = AgglomerativeClustering(n_clusters = 3)

# # Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = ac2.fit_predict(X_principal), cmap ='rainbow')
plt.show()
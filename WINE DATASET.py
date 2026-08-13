# WINE DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


df=pd.read_csv(r"E:\Projects ML and Deep learning\Wine dataset.csv")
X=df.drop("class",axis=1)
Y=df["class"]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print(X_scaled[:5])
pca=PCA()
X_pca=pca.fit_transform(X_scaled)
print("Oringinal shape: ",X_scaled.shape)
print("PCA shape:",X_pca.shape)
print(pca.explained_variance_ratio_)
cumulative_variane=np.cumsum(pca.explained_variance_ratio_)
print(cumulative_variane*100)

pca_2d=PCA(n_components=2)
X_pca_2d=pca_2d.fit_transform(X_scaled)
print("Original shape:",X.shape)
print("Redeuced shape:",X_pca_2d.shape)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca_2d[:,0],
    X_pca_2d[:,1],
    c=Y
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principle Component 2")
plt.title("Wine Dataset-PCA(2 Components)")
plt.show()

kmeans=KMeans(n_clusters=3,random_state=42,n_init=10)
clusters=kmeans.fit_predict(X_pca_2d)
comparison=pd.crosstab(Y,clusters)
print(comparison)
print(clusters)
ari=adjusted_rand_score(Y,clusters)
print("Adjusted Rand Index:",ari)
print(X.shape)
print(Y.shape)
print(df.head())
print(df.info())
print(df.shape)
print(X.describe().T)
print(df.isnull().sum())
print("Dupilcated rows",df.duplicated().sum())
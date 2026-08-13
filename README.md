# 🍷 Wine Classification Analysis using PCA & K-Means Clustering

## 📌 Project Overview

This project applies **Principal Component Analysis (PCA)** and **K-Means Clustering** to the Wine dataset to discover natural groupings among wine samples based on their chemical characteristics.

The dataset contains multiple chemical features, making it a **high-dimensional dataset**. PCA is used to reduce the dimensionality while preserving as much information as possible.

After dimensionality reduction, **K-Means Clustering** is applied to the first two principal components to identify three customer-like groups of wine samples.

Finally, the generated clusters are compared with the actual wine classes using a **Crosstab** and the **Adjusted Rand Index (ARI)**.

---

# 🎯 Project Objectives

The main objectives of this project are:

* Explore the Wine dataset.
* Check the dataset for missing and duplicate values.
* Separate features from the actual class labels.
* Standardize the chemical features.
* Apply PCA for dimensionality reduction.
* Analyze explained variance.
* Reduce the dataset to two principal components.
* Visualize the wine samples in 2D.
* Apply K-Means clustering with 3 clusters.
* Compare the generated clusters with the actual wine classes.
* Evaluate clustering performance using Adjusted Rand Index.

---

# 📊 Dataset

The project uses the **Wine dataset** containing:

* **178 wine samples**
* **13 chemical features**
* **1 class column**

The dataset is loaded using:

```python
df = pd.read_csv(r"E:\Projects ML and Deep learning\Wine dataset.csv")
```

The class column is separated from the feature data:

```python
X = df.drop("class", axis=1)
Y = df["class"]
```

The code therefore uses:

* `X` → chemical features
* `Y` → actual wine class



---

# 🧪 Wine Features

The dataset contains chemical characteristics such as:

* Alcohol
* Malic acid
* Ash
* Alcalinity of ash
* Magnesium
* Total phenols
* Flavanoids
* Nonflavanoid phenols
* Proanthocyanins
* Color intensity
* Hue
* OD280/OD315 of diluted wines
* Proline

These features describe different chemical properties of the wine samples.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn

### Machine Learning Techniques

* Data Exploration
* Feature Scaling
* Principal Component Analysis (PCA)
* K-Means Clustering
* Crosstab Analysis
* Adjusted Rand Index

---

# 🔄 Project Workflow

The complete workflow implemented in the project is:

```text
                  ┌──────────────────────┐
                  │    Load Wine Data    │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │  Explore the Dataset │
                  │ Head / Info / Shape  │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Separate Features    │
                  │ X and Class Y        │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   Standard Scaling   │
                  │    StandardScaler    │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │        PCA           │
                  │ 13 Features → PCs    │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Explained Variance   │
                  │ & Cumulative Variance│
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   PCA to 2D          │
                  │ PC1 + PC2            │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   2D Visualization   │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   K-Means (K=3)      │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Compare with Actual  │
                  │ Classes using Crosstab│
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Adjusted Rand Index  │
                  │       Evaluation     │
                  └──────────────────────┘
```

---

# 1️⃣ Data Exploration

Before applying machine learning, the dataset is inspected.

The code uses:

```python
print(df.head())
print(df.info())
print(df.shape)
print(X.describe().T)
print(df.isnull().sum())
print("Dupilcated rows", df.duplicated().sum())
```

This allows us to examine:

* First few observations
* Dataset structure
* Number of rows and columns
* Statistical summary
* Missing values
* Duplicate rows



---

# 2️⃣ Feature and Target Separation

The class column is separated from the chemical features:

```python
X = df.drop("class", axis=1)
Y = df["class"]
```

### `X`

Contains the **13 chemical features** used for the analysis.

### `Y`

Contains the actual wine class.

An important point is that the class labels are **not used to train K-Means**.

They are used later to compare the discovered clusters with the known classes.

---

# 3️⃣ Feature Scaling

Before applying PCA, the features are standardized:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```



### Why StandardScaler?

The wine features have very different numerical ranges.

For example, one feature might have values around:

```text
1 – 5
```

while another may contain values in the hundreds.

If PCA were applied directly, features with larger numerical scales could have a disproportionate influence.

StandardScaler puts the features onto a comparable scale.

---

# 4️⃣ What is PCA?

**PCA stands for Principal Component Analysis.**

PCA is a **dimensionality reduction technique**.

The Wine dataset contains **13 features**, which makes direct visualization difficult.

We cannot easily visualize:

```text
13-dimensional data
```

But we can visualize:

```text
2-dimensional data
```

PCA helps transform the original features into new variables called **Principal Components**.

```text
Original Features

13 Chemical Features
        ↓
       PCA
        ↓
Principal Components
        ↓
PC1 + PC2
        ↓
     2D Plot
```

---

# 5️⃣ Applying PCA

The project first applies PCA without specifying the number of components:

```python
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
```

This calculates all available principal components.

The original and transformed shapes are then printed:

```python
print("Oringinal shape: ", X_scaled.shape)
print("PCA shape:", X_pca.shape)
```



---

# 6️⃣ Explained Variance

One of the most important concepts in PCA is **Explained Variance**.

The code calculates:

```python
print(pca.explained_variance_ratio_)
```

The explained variance ratio tells us how much of the total information/variance in the dataset is captured by each principal component.

The project obtained:

```text
PC1 → 36.20%
PC2 → 19.21%
```

Together:

```text
PC1 + PC2 ≈ 55.41%
```

The code also calculates cumulative explained variance:

```python
cumulative_variane = np.cumsum(
    pca.explained_variance_ratio_
)

print(cumulative_variane * 100)
```



---

# 7️⃣ PCA for 2D Visualization

After analyzing the explained variance, PCA is specifically reduced to two components:

```python
pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_scaled)
```

The resulting shape is:

```text
178 samples × 2 components
```

The project compares:

```text
Original:
178 × 13

Reduced:
178 × 2
```



This makes the dataset suitable for visualization.

---

# 8️⃣ PCA Visualization

A scatter plot is created using the two principal components:

```python
plt.scatter(
    X_pca_2d[:,0],
    X_pca_2d[:,1],
    c=Y
)
```

The axes represent:

```text
X-axis → Principal Component 1
Y-axis → Principal Component 2
```

The actual wine class is used for coloring the points in this visualization.



This helps us visually investigate whether the known wine classes form distinguishable groups in the reduced feature space.

---

# 9️⃣ K-Means Clustering

After PCA, K-Means clustering is applied to the two PCA components.

The project uses:

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

The clusters are generated using:

```python
clusters = kmeans.fit_predict(X_pca_2d)
```



### Why K = 3?

The Wine dataset contains three known wine classes, and the project applies K-Means with:

```text
n_clusters = 3
```

The class labels are not provided to K-Means during fitting.

---

# 🔟 Understanding K-Means

K-Means attempts to divide the observations into **K clusters**.

The basic process is:

```text
Choose K = 3
      ↓
Initialize centroids
      ↓
Calculate distances
      ↓
Assign points to nearest centroid
      ↓
Recalculate centroids
      ↓
Repeat
      ↓
Final Clusters
```

The final cluster labels are:

```text
0
1
2
```

These numbers do **not automatically correspond** to:

```text
Wine Class 1
Wine Class 2
Wine Class 3
```

Cluster labels are arbitrary.

---

# 1️⃣1️⃣ Comparing Clusters with Actual Classes

After K-Means clustering, the generated cluster assignments are compared with the actual wine classes.

The project uses:

```python
comparison = pd.crosstab(Y, clusters)
print(comparison)
```



The resulting comparison was:

```text
col_0   0   1   2
class
1      59   0   0
2       5   1  65
3       0  48   0
```

This shows how the K-Means clusters correspond to the known wine classes.

For example:

* Class 1 was strongly concentrated in Cluster 0.
* Class 3 was strongly concentrated in Cluster 1.
* Class 2 was mostly concentrated in Cluster 2.

This indicates that the chemical characteristics contain strong structure that K-Means was able to capture.

---

# 1️⃣2️⃣ Adjusted Rand Index

The final evaluation metric used in the project is the **Adjusted Rand Index (ARI)**.

The code calculates:

```python
ari = adjusted_rand_score(Y, clusters)

print("Adjusted Rand Index:", ari)
```



The resulting score was:

```text
Adjusted Rand Index: 0.8950582389649661
```

### What does ARI measure?

ARI measures how closely two clustering assignments agree.

In this project, it compares:

```text
Actual Wine Classes
        VS
K-Means Clusters
```

An ARI closer to **1** indicates stronger agreement.

An ARI of approximately:

```text
0.895
```

indicates a **strong correspondence** between the discovered K-Means clusters and the actual wine classes.

---

# 📈 Final Results

| Metric                           |     Result |
| -------------------------------- | ---------: |
| Samples                          |        178 |
| Original Features                |         13 |
| PCA Components for Visualization |          2 |
| PC1 Variance                     |     36.20% |
| PC2 Variance                     |     19.21% |
| PC1 + PC2 Variance               |     55.41% |
| K-Means Clusters                 |          3 |
| Adjusted Rand Index              | **0.8951** |

---

# 🧠 Key Concepts Learned

## Principal Component Analysis (PCA)

A dimensionality reduction technique that transforms many original features into a smaller number of principal components while retaining important variation in the data.

## Principal Component

A new feature created by PCA from combinations of the original features.

## Explained Variance

Shows how much of the dataset's variation is captured by each principal component.

## StandardScaler

Standardizes features before PCA so that features with larger numerical ranges do not dominate the analysis.

## K-Means Clustering

An unsupervised algorithm that groups observations into a specified number of clusters.

## Crosstab

Used to compare the generated cluster assignments with the known wine classes.

## Adjusted Rand Index

An external clustering evaluation metric used to measure agreement between clustering results and known labels.

---

# ⚠️ Methods Not Used in This Project

To accurately represent the code, this project **does not use**:

* ❌ Elbow Method
* ❌ Silhouette Score
* ❌ Hierarchical Clustering
* ❌ DBSCAN
* ❌ Classification algorithms

The number of K-Means clusters was directly set to **3** in the code. 

---

# 🚀 Possible Future Improvements

The project could be extended by:

* Using the **Elbow Method** to investigate different values of K.
* Using **Silhouette Score** to evaluate cluster quality.
* Comparing K-Means on the original standardized features versus PCA features.
* Applying Hierarchical Clustering.
* Applying DBSCAN.
* Experimenting with different numbers of PCA components.
* Building an interactive visualization using Streamlit.

---

# 📁 Project Structure

```text
Wine-PCA-KMeans/
│
├── WINE DATASET.py
├── Wine dataset.csv
└── README.md
```

---

# 📝 Conclusion

This project demonstrates how **PCA and K-Means can be combined for unsupervised analysis of high-dimensional data**.

The original Wine dataset contains **13 chemical features**. StandardScaler was first used to standardize these features, followed by PCA to transform the data into principal components. The first two components were then used for 2D visualization and K-Means clustering.

K-Means with **3 clusters** produced a strong correspondence with the actual wine classes, achieving an **Adjusted Rand Index of approximately 0.895**.

Overall, the project demonstrates the complete process of:

**Data Exploration → Feature Scaling → PCA → Explained Variance → Dimensionality Reduction → Visualization → K-Means → Cluster Comparison → ARI Evaluation**

---

# 👨‍💻 Author

**Rishabh Jangra**

**Business Analytics | Machine Learning | Data Analytics**

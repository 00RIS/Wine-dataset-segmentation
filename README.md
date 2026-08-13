# 🍷 Wine Dataset — PCA & K-Means Clustering

## 📌 Project Overview

This project applies **Principal Component Analysis (PCA)** and **K-Means Clustering** to the Wine dataset to explore patterns and groupings among different wine samples based on their chemical properties.

The project demonstrates how dimensionality reduction can simplify high-dimensional data and how unsupervised learning can identify natural clusters without using the target class during model training.

## 📊 Dataset

The dataset contains **178 wine samples** with **14 columns**:

* **1 target column:** `class`
* **13 chemical features**

### Features

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

The `class` column represents the known wine category and was kept separate from the unsupervised learning process.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* PCA
* K-Means Clustering
* StandardScaler

## 🔎 Project Workflow

### 1. Data Loading

Loaded the Wine dataset using Pandas.

### 2. Data Inspection

* Checked dataset dimensions
* Examined data types
* Checked for missing values
* Checked for duplicate records

Result:

* **178 samples**
* **13 input features**
* **0 missing values**
* **0 duplicate rows**

### 3. Feature and Target Separation

```python
X = df.drop("class", axis=1)
y = df["class"]
```

`X` contains the chemical features, while `y` contains the actual wine classes.

The class labels were **not used during PCA or K-Means clustering**.

### 4. Feature Scaling

Standardization was performed using `StandardScaler` because the features have different numerical ranges.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. Principal Component Analysis

PCA was applied to reduce the dimensionality of the dataset.

The first two principal components explained:

* **PC1:** 36.20%
* **PC2:** 19.21%
* **PC1 + PC2:** 55.41%

For approximately 90% variance retention:

* **7 components:** 89.34%
* **8 components:** 92.02%

For visualization, the first two components were selected.

### 6. PCA Visualization

The original **13-dimensional dataset** was reduced to **2 dimensions** using PCA.

This allowed the wine samples to be visualized on a 2D scatter plot.

The visualization showed approximately **three distinct groups**.

### 7. K-Means Clustering

K-Means was applied to the two PCA components using:

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

The model generated three clusters.

### 8. Cluster Evaluation

The resulting clusters were compared with the actual wine classes using a **confusion-style cross-tabulation** and the **Adjusted Rand Index (ARI)**.

The resulting ARI was:

```text
0.8951
```

An ARI close to 1 indicates strong agreement between the discovered clusters and the known class grouping.

## 📈 Key Results

| Metric                           |     Result |
| -------------------------------- | ---------: |
| Samples                          |        178 |
| Original Features                |         13 |
| PCA Components for Visualization |          2 |
| Variance Explained by PC1        |     36.20% |
| Variance Explained by PC1 + PC2  |     55.41% |
| Components for ~90% Variance     |          8 |
| Number of K-Means Clusters       |          3 |
| Adjusted Rand Index              | **0.8951** |

## 🧠 Key Learnings

* PCA reduces dimensionality while preserving important variance.
* Feature scaling is important before PCA.
* Principal components are new features created from combinations of the original features.
* PCA itself does not perform classification or clustering.
* K-Means can discover groups without using class labels.
* Cluster labels are arbitrary and do not necessarily match class numbers.
* ARI can be used to evaluate how closely clustering results match known groupings.

## 🚀 Future Improvements

* Experiment with different numbers of K-Means clusters.
* Compare clustering using the original features versus PCA features.
* Apply Hierarchical Clustering and DBSCAN.
* Explore different PCA component selections.
* Use additional clustering evaluation metrics such as Silhouette Score.

## 📁 Project Structure

```text
Wine-PCA-KMeans/
│
├── WINE_DATASET.py
├── wine.csv
└── README.md
```

## 👨‍💻 Author

**Rishabh Jangra**

Business Analytics | Machine Learning | Data Analytics

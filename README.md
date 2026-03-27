# 🗺️ Bangladesh-Crime-Clustering

A full end-to-end unsupervised machine learning pipeline for clustering crime incidents across Bangladesh into distinct regional groups — supporting **data-driven policing strategies** for the Ministry of Home Affairs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Optimal K-Selection & Modeling](#optimal-k-selection--modeling)
- [Evaluation](#evaluation)
- [Cluster Profiling](#cluster-profiling)
- [Cluster Visualization](#cluster-visualization)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## 📌 Overview

This project tackles an **unsupervised clustering problem** to discover natural groupings of crime incidents across Bangladesh based on demographic, infrastructural, and environmental features. Unlike classification, there is no predefined label — the `crime` column is used **exclusively for external validation**, never for training.

**Discovered Clusters:**

| Cluster | Name | Characteristic |
|---------|------|----------------|
| 0 | Urban Megacity | High density, high literacy, strong infrastructure |
| 1 | Rural Frontier | Low density, low literacy, severely under-resourced |

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| File | `Bangladesh Crime.csv` |
| Shape | 6,574 rows × 25 columns (before cleaning) |
| Target Column | `crime` — **reference only**, not used in modeling |

**Feature Categories:**

| Category | Features |
|----------|---------|
| Temporal | `incident_month`, `incident_week`, `incident_weekday`, `weekend`, `part_of_the_day`, `season` |
| Geographic | `incident_district`, `incident_division` |
| Weather | `precip`, `visibility`, `heatindex` |
| Demographic | `total_population`, `gender_ration`, `average_household_size`, `density_per_kmsq`, `literacy_rate` |
| Infrastructure | `religious_institution`, `playground`, `park`, `police_station`, `school`, `college` |

**Columns Dropped (non-informative or redundant):**
`Unnamed: 0`, `incident_district` (high cardinality), `male_population`, `female_population` (redundant with `total_population`)

---

## 🗂 Project Structure

```
bangladesh-crime-clustering/
│
├── Bangladesh Crime.csv      # Raw dataset
├── notebook.ipynb            # Main analysis notebook
└── README.md
```

---

## 🔧 Pipeline Walkthrough

### 1. 🧹 Data Preprocessing & Cleaning

- **Missing Values**:
  - `part_of_the_day` (1.78%) → drop rows (small, safe to discard)
  - `precip`, `literacy_rate` → median imputation (preserve distribution)
  - `season` → mode imputation (dominant category fills gap)
- **Duplicates**: Identified and removed
- **Inconsistency Fix**: `incident_weekday` had 400+ unique values due to typos — standardized using **fuzzy matching** (`rapidfuzz`) back to 7 clean day names
- **Outliers**:
  - Logical floor clip for infrastructure columns (`religious_institution`, `police_station`, etc.) — values like `-99` and `-1` corrected to `0`
  - Winsorization at 1st–99th percentile for skewed continuous features (`precip`, `total_population`, `density_per_kmsq`, etc.)

### 2. 📊 Exploratory Data Analysis (EDA)

- **Crime Distribution**: `bodyfound` and `murder` nearly tied at 23.5% vs 23.0% — highlighted with dotted bracket annotation
- **KDE Plots**: Distribution per numerical feature with median reference line and area fill
- **Correlation Heatmap**: Pairwise correlation across all numerical features
- **Categorical Distributions**: Dominant category highlighted per feature with accent color

### 3. ⚙️ Preprocessing Pipeline

All features are processed through a scikit-learn `ColumnTransformer` before clustering:

| Feature Type | Transformer |
|---|---|
| Numerical + Binary | `StandardScaler` — mandatory for distance-based K-Means |
| Categorical | `OneHotEncoder(handle_unknown='ignore')` |

> **Why scaling is mandatory:** K-Means computes Euclidean distance. Without scaling, `density_per_kmsq` (~30,000) would completely dominate `park` (~3), making those features invisible to the algorithm.

---

## 🤖 Optimal K-Selection & Modeling

Four metrics evaluated across **K=2 to K=10** to determine the optimal number of clusters:

| Metric | Direction | Role |
|--------|-----------|------|
| Inertia (Elbow) | Lower is better | Structural fit |
| Silhouette Score | Higher is better | Cluster separation quality |
| Davies-Bouldin | Lower is better | Cluster compactness |
| Calinski-Harabasz | Higher is better | Between-vs-within variance ratio |

**Optimal K = 2** selected based on highest Silhouette Score and lowest Davies-Bouldin Score.

**Final model configuration:**
```python
KMeans(
    n_clusters=2,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
```

---

## 📈 Evaluation

Cluster quality assessed using internal validation metrics on the final K=2 model:

| Metric | Direction |
|--------|-----------|
| Silhouette Score | ↑ Higher is better (max 1.0) |
| Davies-Bouldin | ↓ Lower is better |
| Calinski-Harabasz | ↑ Higher is better |
| Inertia | ↓ Lower is better |

> No accuracy or F1 score — this is unsupervised learning. There is no ground truth label to evaluate against.

---

## 🔍 Cluster Profiling

Cluster "DNA" extracted by aggregating features per cluster (median for numerics, mode for categoricals):

| Feature | Cluster 0 — Urban Megacity | Cluster 1 — Rural Frontier |
|---------|---------------------------|---------------------------|
| `density_per_kmsq` | ~30,551 | ~679 |
| `literacy_rate` | ~74.6% | ~52.9% |
| `police_station` | ~60 | ~4 |
| `total_population` | ~8.9M | ~376K |
| `dominant crime` | Body Found | Murder |
| `dominant division` | Dhaka (core) | Dhaka (outer) |

**Centroid heatmap** uses z-score normalization against full dataset mean/std — ensuring features with different scales are visually comparable.

**Key Insight:** A critical resource imbalance exists. Cluster 0 averages ~60 police stations, while Cluster 1 has only ~4 — correlating with higher rates of direct violent crime (murder) in under-resourced areas.

---

## 🗺️ Cluster Visualization

| Method | Purpose |
|--------|---------|
| **PCA 2D** | Linear dimensionality reduction — shows global cluster separation with centroids projected |
| **t-SNE 2D** | Non-linear reduction — captures local cluster topology |
| **External Validation** | Cross-tab of `crime` type per cluster — confirms clusters align with meaningfully different crime patterns, even though `crime` was never used in training |

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn rapidfuzz
```

> Python 3.10+ recommended.

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/werrenedbert06/Bangladesh-Crime-Clustering.git
   cd Bangladesh-Crime-Clustering
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn rapidfuzz
   ```

3. **Place the dataset**

   Ensure `Bangladesh Crime.csv` is in the root directory.

4. **Run the notebook**
   ```bash
   jupyter notebook notebook.ipynb
   ```

---

## 🧠 Key Design Decisions

- **No train/test split**: K-Means is unsupervised — all data is used for fitting. There is no prediction task that would require held-out data.
- **`crime` as external validation only**: The label exists in the dataset but is deliberately excluded from the feature matrix. It is used post-clustering to validate whether the discovered groups align with crime type distributions.
- **Clip before Winsorize**: Negative infrastructure values (`-99`, `-1`) are fixed via logical floor clip first — these are data entry errors, not statistical outliers. Winsorization handles distributional extremes separately.
- **Fuzzy matching for weekday**: `incident_weekday` had 400+ unique values due to inconsistent input. RapidFuzz maps every variant to the closest of the 7 correct day names.
- **Z-score normalization for heatmap**: Raw centroid values are not comparable across features (`density_per_kmsq` ~30k vs `park` ~3). Normalizing against full-dataset statistics makes the heatmap meaningful.

---

## 📄 License

This project is for educational and portfolio purposes. Feel free to fork and adapt.

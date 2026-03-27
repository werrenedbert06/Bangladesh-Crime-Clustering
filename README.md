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
| Geographic | `incident_division` |
| Weather | `precip`, `visibility`, `heatindex` |
| Demographic | `total_population`, `gender_ration`, `average_household_size`, `density_per_kmsq`, `literacy_rate` |
| Infrastructure | `religious_institution`, `playground`, `park`, `police_station`, `school`, `college` |
| Binary | `weekend` |

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

- **Drop columns**: `Unnamed: 0`, `incident_district`, `male_population`, `female_population`
- **Feature groups defined**: `NUM_FEATURES` (16), `CAT_FEATURES` (4), `BIN_FEATURES` (1)
- **Missing values**:
  - `precip`, `literacy_rate` → median imputation
  - `season` → mode imputation
  - `part_of_the_day` (1.78%) → drop rows (small, safe to discard)
- **Duplicates**: identified and removed
- **Inconsistency fix**: `incident_weekday` had 400+ unique values due to typos — standardized using **fuzzy matching** (`rapidfuzz`) back to 7 clean day names
- **Outliers**:
  - Logical floor clip (`religious_institution`, `playground`, `police_station`, `park`, `school`, `college`) — values like `-99` and `-1` corrected to `0`
  - Winsorization at 1st–99th percentile for skewed continuous features (`precip`, `total_population`, `gender_ration`, `average_household_size`, `density_per_kmsq`)

### 2. 📊 Exploratory Data Analysis (EDA)

- **Crime Distribution**: `bodyfound` and `murder` nearly tied — highlighted with dotted bracket annotation showing gap
- **KDE Plots**: Distribution for 8 key numerical features with median dashed line and shaded area fill
- **Correlation Heatmap**: Pairwise correlation across all `NUM_FEATURES`
- **Categorical Distributions**: 2×2 grid, dominant category highlighted with accent color per feature

### 3. ⚙️ Preprocessing Pipeline

All features processed through a scikit-learn `ColumnTransformer` before clustering:

| Feature Type | Transformer |
|---|---|
| Numerical + Binary | `StandardScaler` — mandatory for distance-based K-Means |
| Categorical | `OneHotEncoder(handle_unknown='ignore')` |

> **Why scaling is mandatory:** K-Means computes Euclidean distance. Without scaling, `density_per_kmsq` (~30,000) would completely dominate `park` (~3), making those features invisible to the algorithm.

---

## 🤖 Optimal K-Selection & Modeling

Four metrics evaluated across **K=2 to K=10**:

| Metric | Direction | Role |
|--------|-----------|------|
| Inertia (Elbow) | Lower is better | Structural fit |
| Silhouette Score | Higher is better | Cluster separation quality |
| Davies-Bouldin | Lower is better | Cluster compactness |
| Calinski-Harabasz | Higher is better | Between-vs-within variance ratio |

Results printed as a formatted table, then visualized as a **3-panel chart** (Elbow / Silhouette / DB+CH dual-axis).

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
| Inertia | ↓ Lower is better |
| Silhouette Score | ↑ Higher is better (max 1.0) |
| Davies-Bouldin | ↓ Lower is better |
| Calinski-Harabasz | ↑ Higher is better |

> No accuracy or F1 score — this is unsupervised learning. There is no ground truth label to evaluate against.

---

## 🔍 Cluster Profiling

### Aggregation
Cluster "DNA" extracted by groupby aggregation:
- **Numerical** (`PROFILE_NUM`): median per cluster
- **Categorical** (`PROFILE_CAT`): mode per cluster — includes `crime` as **reference only**

### Cluster Size Distribution
Bar chart showing proportion of data in each cluster.

### Centroid Heatmap
Normalized using z-score against full dataset mean/std:
```python
profile_norm = (num_profile - df[PROFILE_NUM].mean()) / df[PROFILE_NUM].std()
```
Ensures features with different scales (`density_per_kmsq` ~30k vs `park` ~3) are visually comparable. Red = above dataset average, Blue = below.

### Feature Distribution per Cluster
Boxplots for 6 key discriminating features: `density_per_kmsq`, `literacy_rate`, `police_station`, `total_population`, `school`, `heatindex`.

### Cluster Names

| Cluster | Assigned Name |
|---------|--------------|
| 0 | Urban Megacity (High Density) |
| 1 | Rural Frontier (Low Resource) |

**Key Insight:** Cluster 0 averages ~60 police stations vs Cluster 1's ~4 — a 15× gap in law enforcement coverage correlating with higher rates of direct violent crime (murder) in under-resourced areas.

---

## 🗺️ Cluster Visualization

| Method | Purpose |
|--------|---------|
| **PCA 2D** | Linear dimensionality reduction — global cluster separation with centroids (✕) projected into PCA space |
| **t-SNE 2D** | Non-linear reduction — captures local cluster topology |
| **External Validation (cross-tab)** | Crime type proportion per cluster — confirms clusters align with different crime patterns, even though `crime` was never used in training |
| **Stacked Bar** | Visual representation of crime type distribution per cluster |

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
- **`crime` as external validation only**: The label exists in the dataset but is deliberately excluded from the feature matrix. It is used post-clustering to confirm whether discovered groups align with different crime type distributions.
- **Clip before Winsorize**: Negative infrastructure values (`-99`, `-1`) are fixed via logical floor clip first — these are data entry errors, not statistical outliers. Winsorization handles distributional extremes separately.
- **Fuzzy matching for weekday**: `incident_weekday` had 400+ unique values due to inconsistent input. RapidFuzz maps every variant to the nearest of the 7 correct day names.
- **Z-score normalization for centroid heatmap**: Raw centroid values are not comparable across features. Normalizing against full-dataset mean/std makes relative positions across clusters visually meaningful.

---

## 📄 License

This project is for educational and portfolio purposes. Feel free to fork and adapt.

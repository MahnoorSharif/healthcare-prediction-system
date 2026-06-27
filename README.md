# Healthcare Risk Prediction System
**CSL-487 Data Science Lab | Bahria University | Spring 2026**
Lab Engineer: Ms. Summaiya Mehmood

## Group Members
- Mahnoor — 02-235232-001
- Hadiqa Mehmood — 02-235232-007
- Kanwer Farasat Ali — 02-235232-012

---

## Folder Structure
```
healthcare_project/
│
├── datasets/
│   ├── diabetes.csv          ← Pima Indians Diabetes Dataset (768 rows)
│   └── heart.csv             ← Heart Disease Dataset (1025 rows)
│
├── notebooks/
│   └── healthcare_analysis.ipynb   ← ALL phases: Analysis, Preprocessing, EDA, PCA, Models
│
├── models/                   ← Auto-generated after running notebook
│   ├── scaler_diabetes.pkl
│   ├── scaler_heart.pkl
│   ├── heart_columns.pkl
│   ├── pca_diabetes.pkl
│   ├── pca_heart.pkl
│   ├── logistic_regression_diabetes.pkl
│   ├── knn_diabetes.pkl
│   ├── decision_tree_diabetes.pkl
│   ├── logistic_regression_heart.pkl
│   ├── knn_heart.pkl
│   ├── decision_tree_heart.pkl
│   ├── best_models.json
│   └── plot_*.png            ← All saved visualizations
│
├── app.py                    ← Streamlit Dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Jupyter Notebook FIRST
Open VS Code → open `notebooks/healthcare_analysis.ipynb` → Run All Cells
This will:
- Perform data analysis on both datasets
- Preprocess the data (no rows dropped)
- Generate all EDA visualizations
- Train all 3 ML models
- Save models and scalers to /models folder

### Step 3 — Launch the FastAPI Backend
Use the local repository virtual environment when possible. From `D:\DS LAB PROJECT\healthcare_project` run:

```powershell
..\.venv\Scripts\python.exe -m uvicorn backend:app --reload --port 8000
```

If you are already in `healthcare_project` and you see errors from `scikit-learn`, do not run `d:\anaconda\python.exe`; instead use the local venv Python above.

### Step 4 — Run the React Frontend
```bash
cd healthcare_project/frontend
npm install
npm run dev
```
Opens at: http://localhost:5173

> The React dashboard uses the FastAPI backend for predictions; start the backend first, then open the frontend.

---

## Project Phases (Notebook)
| Phase | Content |
|-------|---------|
| 1 | Library imports |
| 2 | Data Analysis — head, tail, shape, size, columns, dtypes, info, describe, nulls, target dist |
| 3 | Preprocessing — median imputation, IQR capping, one-hot encoding, StandardScaler, train/test split |
| 4 | EDA Visualizations — histograms, boxplots, heatmaps, pairplots, violin plots |
| 5 | PCA — explained variance, 2D scatter |
| 6 | Model Training — Logistic Regression, KNN, Decision Tree |
|   | Evaluation — accuracy, precision, recall, F1, confusion matrix, ROC curve, feature importance |

## Dashboard Tabs
| Tab | Content |
|-----|---------|
| 🔮 Prediction | Disease selector, patient input form, 3-model comparison, HIGH/LOW RISK result |
| 📊 EDA | All visualizations for both datasets |
| 📈 Model Metrics | Metrics table, confusion matrices, ROC curves, feature importance |
| 🔵 PCA | Explained variance bar, 2D scatter by class |

## 🌐 Multilingual Support
Supports **English and Urdu** with full RTL layout.
Toggle language from the top navigation bar.

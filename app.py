"""
Healthcare Risk Prediction System — Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'datasets')
MDL_DIR  = os.path.join(BASE, 'models')

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 1rem; padding-left:1.5rem; padding-right:1.5rem; }
    .stApp { background: linear-gradient(180deg, #07101f 0%, #0e1a2d 100%); color: #eef3ff; }
    .main-title   { font-size:2.4rem; font-weight:700; color:#8ab4f8; text-align:center; margin-bottom:0.2rem; }
    .sub-title    { font-size:1rem; color:#cfd8ff; text-align:center; margin-bottom:1.5rem; }
    .hero-banner { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.16); border-radius: 24px; padding: 1.3rem; margin-bottom: 1.4rem; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom:1.5rem; }
    .summary-card { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 18px; padding: 1rem 1.2rem; color: #eef3ff; }
    .summary-card strong { display: block; font-size: 1.5rem; margin-bottom: 0.25rem; }
    .summary-card span { color: #c8d7ff; }
    .dashboard-card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: 20px; padding: 1.2rem; margin-bottom: 1.2rem; }
    .dashboard-card h3 { margin-bottom: 0.5rem; color: #eef3ff; }
    .stButton>button, .stDownloadButton>button { border-radius: 14px; border: none; padding: 0.85rem 1rem; font-weight: 600; }
    .stButton>button { background-color: #4f8cf7; color: #fff; }
    .stButton>button:hover, .stDownloadButton>button:hover { opacity: 0.94; }
    .stAlert { background-color: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.16) !important; color: #eef3ff !important; }
    .section-head { font-size:1.3rem; font-weight:700; color:#b6c6ff; margin-bottom:0.7rem; }
    .risk-high { background:#3c1f23; border-left:6px solid #f06257; padding:1rem; border-radius:12px; font-size:1.15rem; font-weight:700; color:#ffcdd2; }
    .risk-low { background:#1e2f1f; border-left:6px solid #66bb6a; padding:1rem; border-radius:12px; font-size:1.15rem; font-weight:700; color:#dcedc8; }
    .best-badge { background:#d8eaff; border:1px solid #7da7ff; border-radius:8px; padding:3px 10px; font-size:0.85rem; color:#1f3f8c; }
    .welcome-card { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); border-radius: 28px; padding: 2rem; max-width: 930px; margin: 2rem auto; text-align: center; }
    .welcome-card h1 { color: #ffffff; margin-bottom: 0.5rem; }
    .welcome-card p { color: #c7d2ff; font-size:1.03rem; line-height:1.7; }
    .welcome-pill { background: rgba(255,255,255,0.12); color: #a8b8ff; border-radius: 999px; padding: 0.55rem 1rem; display: inline-block; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Load assets ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    assets = {}
    try:
        # Diabetes
        assets['scaler_d']   = joblib.load(os.path.join(MDL_DIR, 'scaler_diabetes.pkl'))
        assets['lr_d']       = joblib.load(os.path.join(MDL_DIR, 'logistic_regression_diabetes.pkl'))
        assets['knn_d']      = joblib.load(os.path.join(MDL_DIR, 'knn_diabetes.pkl'))
        assets['dt_d']       = joblib.load(os.path.join(MDL_DIR, 'decision_tree_diabetes.pkl'))
        assets['pca_d']      = joblib.load(os.path.join(MDL_DIR, 'pca_diabetes.pkl'))
        # Heart
        assets['scaler_h']   = joblib.load(os.path.join(MDL_DIR, 'scaler_heart.pkl'))
        assets['lr_h']       = joblib.load(os.path.join(MDL_DIR, 'logistic_regression_heart.pkl'))
        assets['knn_h']      = joblib.load(os.path.join(MDL_DIR, 'knn_heart.pkl'))
        assets['dt_h']       = joblib.load(os.path.join(MDL_DIR, 'decision_tree_heart.pkl'))
        assets['pca_h']      = joblib.load(os.path.join(MDL_DIR, 'pca_heart.pkl'))
        assets['heart_cols'] = joblib.load(os.path.join(MDL_DIR, 'heart_columns.pkl'))
        # Best model info
        with open(os.path.join(MDL_DIR, 'best_models.json')) as f:
            assets['best'] = json.load(f)
        assets['loaded'] = True
    except Exception as e:
        assets['loaded'] = False
        assets['error']  = str(e)
    return assets

@st.cache_data
def load_data():
    df_d = pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))
    df_h = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))
    return df_d, df_h

assets = load_assets()
df_d, df_h = load_data()

def clean_model_input(X):
    """Make sure sklearn models never receive NaN or infinite values."""
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True

if st.session_state.show_welcome:
    st.markdown(
        "<div class='welcome-card'>"
        "<div class='welcome-pill'>Welcome to the Healthcare Risk Dashboard</div>"
        "<h1>Predict Diabetes & Heart Disease Risk</h1>"
        "<p>Use fast, explainable ML models and interactive visualizations to explore patient risk profiles, feature impacts, and PCA insights.</p>"
        "</div>",
        unsafe_allow_html=True
    )
    if st.button('🚀 Enter Dashboard'):
        with st.spinner('Loading dashboard...'):
            time.sleep(0.75)
        st.session_state.show_welcome = False
        st.experimental_rerun()
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/7/7a/Bahria_University_logo.svg/200px-Bahria_University_logo.svg.png",
             width=120)
    st.markdown("### 🏥 Healthcare Risk Prediction")
    st.markdown("**CSL-487 | Spring 2026**")
    st.markdown("---")
    disease = st.selectbox("🔬 Select Disease", ["Diabetes", "Heart Disease"],
                           help="Choose which disease to predict")
    st.markdown("---")
   

# ─── Main Title ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 Healthcare Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Predicting <b>{disease}</b> risk using Machine Learning | Bahria University</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-banner">'
    '<div class="summary-grid">'
    '<div class="summary-card"><strong>2 Datasets</strong><span>Diabetes and Heart Disease risk prediction.</span></div>'
    '<div class="summary-card"><strong>3 Models</strong><span>Logistic Regression, KNN, Decision Tree.</span></div>'
    '<div class="summary-card"><strong>Interactive Insights</strong><span>Visual analytics, metrics, and PCA.</span></div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)
# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 EDA & Visualizations", "📈 Model Metrics", "🔵 PCA Analysis"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not assets['loaded']:
        st.error(f"⚠️ Models not loaded. Please run the notebook first!\n\nError: {assets.get('error','')}")
        st.stop()

    st.markdown('<div class="section-head">👤 Enter Patient Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-grid">'
        '<div class="summary-card"><strong>Fast Predictions</strong><span>Generate a risk score in seconds.</span></div>'
        '<div class="summary-card"><strong>Explainable Models</strong><span>Compare Logistic, KNN, and Decision Tree outputs.</span></div>'
        '<div class="summary-card"><strong>Clean Inputs</strong><span>Interactive sliders and selectors for each patient feature.</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

    if disease == "Diabetes":
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies  = st.slider("Pregnancies",          0, 17, 3)
            glucose      = st.slider("Glucose (mg/dL)",      50, 200, 120)
            bp           = st.slider("Blood Pressure",        40, 130, 70)
        with col2:
            skin         = st.slider("Skin Thickness (mm)",   0,  99, 20)
            insulin      = st.slider("Insulin (µU/mL)",        0, 846, 80)
            bmi          = st.slider("BMI",                  15.0, 67.0, 25.0, step=0.1)
        with col3:
            dpf          = st.slider("Diabetes Pedigree Fn", 0.0,  2.5, 0.5, step=0.01)
            age          = st.slider("Age",                  21,  81,  33)

        raw_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled_input = clean_model_input(assets['scaler_d'].transform(raw_input))

        model_map = {
            'Logistic Regression': assets['lr_d'],
            'KNN':                 assets['knn_d'],
            'Decision Tree':       assets['dt_d'],
        }
        best_name = assets['best']['diabetes']
        labels    = ['No Diabetes', 'Diabetes']

    elif disease == 'Heart Disease':  # Heart Disease
        col1, col2, col3 = st.columns(3)
        with col1:
            age_h     = st.slider("Age",              29,  77,  54)
            sex       = st.selectbox("Sex",           ["Male (1)", "Female (0)"])
            sex_val   = 1 if "Male" in sex else 0
            cp        = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
            trestbps  = st.slider("Resting BP (mmHg)", 90, 200, 130)
        with col2:
            chol      = st.slider("Cholesterol (mg/dL)", 120, 570, 240)
            fbs       = st.selectbox("Fasting Blood Sugar > 120", ["No (0)", "Yes (1)"])
            fbs_val   = 1 if "Yes" in fbs else 0
            restecg   = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
            thalach   = st.slider("Max Heart Rate", 70, 202, 150)
        with col3:
            exang     = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
            exang_val = 1 if "Yes" in exang else 0
            oldpeak   = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
            slope     = st.selectbox("Slope of ST segment", [0, 1, 2])
            ca        = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4])
            thal      = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

        # Build input matching one-hot encoded columns
        base_input = {
            'age': age_h, 'sex': sex_val, 'fbs': fbs_val, 'trestbps': trestbps,
            'chol': chol, 'thalach': thalach, 'exang': exang_val, 'oldpeak': oldpeak
        }
        # One-hot for cp, restecg, slope, ca, thal
        for col_name, val, cats in [
            ('cp', cp, [0,1,2,3]),
            ('restecg', restecg, [0,1,2]),
            ('slope', slope, [0,1,2]),
            ('ca', ca, [0,1,2,3,4]),
            ('thal', thal, [0,1,2,3]),
        ]:
            for c in cats:
                base_input[f'{col_name}_{c}'] = 1 if val == c else 0

        heart_cols = assets['heart_cols']
        input_df   = pd.DataFrame([{c: base_input.get(c, 0) for c in heart_cols}])
        scaled_input = clean_model_input(assets['scaler_h'].transform(input_df))

        model_map = {
            'Logistic Regression': assets['lr_h'],
            'KNN':                 assets['knn_h'],
            'Decision Tree':       assets['dt_h'],
        }
        best_name = assets['best']['heart']
        labels    = ['No Disease', 'Heart Disease']

    # ── Run prediction ──
    st.markdown("---")
    if st.button("🔮 Predict Risk", width="stretch", type="primary"):
        results_cols = st.columns(3)
        predictions  = {}

        for i, (mname, model) in enumerate(model_map.items()):
            pred     = model.predict(scaled_input)[0]
            prob     = model.predict_proba(scaled_input)[0][1] * 100
            predictions[mname] = {'pred': pred, 'prob': prob}
            risk_str = "🔴 HIGH RISK" if pred == 1 else "🟢 LOW RISK"

            is_best = " ⭐ Best" if mname == best_name else ""
            with results_cols[i]:
                st.markdown(f"**{mname}**{is_best}")
                st.metric("Prediction", risk_str)
                st.metric("Probability", f"{prob:.1f}%")

        # Final result from best model
        st.markdown("---")
        best_pred = predictions[best_name]['pred']
        best_prob = predictions[best_name]['prob']

        if best_pred == 1:
            st.markdown(f'<div class="risk-high">🔴 FINAL RESULT: HIGH RISK of {disease}<br>'
                        f'<span style="font-size:1rem;font-weight:400;">Confidence: {best_prob:.1f}% | Based on: {best_name}</span></div>',
                        unsafe_allow_html=True)
            st.warning("⚠️ Please consult a healthcare professional immediately.")
        else:
            st.markdown(f'<div class="risk-low">🟢 FINAL RESULT: LOW RISK of {disease}<br>'
                        f'<span style="font-size:1rem;font-weight:400;">Confidence: {100-best_prob:.1f}% healthy | Based on: {best_name}</span></div>',
                        unsafe_allow_html=True)
            st.success("✅ Patient appears to be at low risk. Maintain a healthy lifestyle!")

        # Model comparison bar
        st.markdown("#### 📊 Model Probability Comparison")
        fig, ax = plt.subplots(figsize=(8, 3))
        mnames = list(predictions.keys())
        probs  = [predictions[m]['prob'] for m in mnames]
        colors = ['#F44336' if p >= 50 else '#4CAF50' for p in probs]
        bars = ax.barh(mnames, probs, color=colors, edgecolor='white', height=0.5)
        ax.axvline(50, color='gray', linestyle='--', linewidth=1.5)
        for bar, val in zip(bars, probs):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontweight='bold')
        ax.set_xlim(0, 110)
        ax.set_xlabel('Risk Probability (%)')
        ax.set_title('Model Predictions Comparison')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    eda_ds = st.radio("Select Dataset", ["Diabetes", "Heart Disease"], horizontal=True)

    if eda_ds == "Diabetes":
        df_plot = df_d.copy()
        target  = 'Outcome'
        num_cols = df_d.columns.tolist()
        class_map = {0: 'No Diabetes', 1: 'Diabetes'}
        colors2   = ['#4CAF50', '#F44336']
    else:
        # Heart Disease
        df_plot  = df_h.copy()
        target   = 'target'
        num_cols = ['age','trestbps','chol','thalach','oldpeak','target']
        class_map = {0: 'No Disease', 1: 'Heart Disease'}
        colors2   = ['#2196F3', '#FF5722']

    eda_opt = st.selectbox("Choose Visualization", [
        "Class Distribution",
        "Feature Histograms",
        "Boxplots",
        "Correlation Heatmap",
        "Violin Plots",
        "Pairplot (Key Features)"
    ])

    fig = None

    if eda_opt == "Class Distribution":
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = df_plot[target].value_counts()
        bars = ax.bar([class_map[k] for k in vals.index], vals.values,
                      color=colors2, edgecolor='white', linewidth=2)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(v), ha='center', fontweight='bold')
        ax.set_title(f'{eda_ds} — Class Distribution', fontweight='bold')
        ax.set_ylabel('Count')

    elif eda_opt == "Feature Histograms":
        cols_to_plot = [c for c in df_plot.columns if c != target]
        n = len(cols_to_plot)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            axes[i].hist(df_plot[col], bins=25, color=colors2[0], edgecolor='white', alpha=0.85)
            axes[i].set_title(col, fontweight='bold')
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f'{eda_ds} — Feature Histograms', fontsize=14, fontweight='bold')
        plt.tight_layout()

    elif eda_opt == "Boxplots":
        cols_to_plot = [c for c in df_plot.columns if c != target]
        ncols = 4
        nrows = (len(cols_to_plot) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
        axes = axes.flatten()
        df_plot[target] = df_plot[target].map(class_map)
        for i, col in enumerate(cols_to_plot):
            sns.boxplot(data=df_plot, x=target, y=col, palette=colors2, ax=axes[i])
            axes[i].set_title(col, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=10)
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(f'{eda_ds} — Boxplots by Class', fontsize=14, fontweight='bold')
        plt.tight_layout()

    elif eda_opt == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df_plot[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, linewidths=0.5, ax=ax, square=True)
        ax.set_title(f'{eda_ds} — Correlation Heatmap', fontweight='bold')

    elif eda_opt == "Violin Plots":
        if eda_ds == "Diabetes":
            vio_cols = ['Glucose','BMI','Age','BloodPressure','Insulin']
        else:
            # Heart Disease
            vio_cols = ['age','trestbps','chol','thalach','oldpeak']
        fig, axes = plt.subplots(1, len(vio_cols), figsize=(18, 5))
        df_v = df_plot.copy()
        if target in df_v.columns and df_v[target].dtype != object:
            df_v[target] = df_v[target].map(class_map)
        for i, col in enumerate(vio_cols):
            sns.violinplot(data=df_v, x=target, y=col,
                           palette=colors2, ax=axes[i], inner='quartile')
            axes[i].set_title(col, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=15)
        plt.suptitle(f'{eda_ds} — Violin Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()

    elif eda_opt == "Pairplot (Key Features)":
        if eda_ds == "Diabetes":
            key_cols = ['Glucose','BMI','Age','DiabetesPedigreeFunction', target]
        else:
            # Heart Disease
            key_cols = ['age','trestbps','chol','thalach', target]
        df_pp = df_plot[key_cols].copy()
        if df_pp[target].dtype != object:
            df_pp[target] = df_pp[target].map(class_map)
        pal = {list(class_map.values())[0]: colors2[0], list(class_map.values())[1]: colors2[1]}
        pp = sns.pairplot(df_pp, hue=target, palette=pal, plot_kws={'alpha':0.5}, diag_kind='kde')
        pp.fig.suptitle(f'{eda_ds} — Pairplot', fontsize=13, fontweight='bold', y=1.01)
        st.pyplot(pp.fig)
        plt.close('all')
        fig = None

    if fig is not None:
        st.pyplot(fig)
        plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not assets['loaded']:
        st.error("⚠️ Run the notebook first to generate models!")
    else:
        st.markdown('<div class="section-head">📈 Model Evaluation Metrics</div>', unsafe_allow_html=True)
        metrics_ds = st.radio("Dataset", ["Diabetes", "Heart Disease"], horizontal=True, key='metrics_ds')

        # Re-prepare test data
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import warnings; warnings.filterwarnings('ignore')

        if metrics_ds == "Diabetes":
            df_proc = df_d.copy()
            zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
            df_proc[zero_cols] = df_proc[zero_cols].replace(0, np.nan)
            for col in zero_cols:
                df_proc[col].fillna(df_proc[col].median(), inplace=True)
            for col in [c for c in df_proc.columns if c != 'Outcome']:
                Q1, Q3 = df_proc[col].quantile(0.25), df_proc[col].quantile(0.75)
                IQR = Q3 - Q1
                df_proc[col] = df_proc[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            X = clean_model_input(assets['scaler_d'].transform(df_proc.drop('Outcome', axis=1)))
            y = df_proc['Outcome'].values
            _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            mods = {'Logistic Regression': assets['lr_d'],
                    'KNN': assets['knn_d'],
                    'Decision Tree': assets['dt_d']}
            best_name = assets['best']['diabetes']
            class_labels = ['No Diabetes', 'Diabetes']
        elif metrics_ds == 'Heart Disease':
            df_proc = df_h.copy()
            num_h = ['age','trestbps','chol','thalach','oldpeak']
            for col in num_h:
                Q1, Q3 = df_proc[col].quantile(0.25), df_proc[col].quantile(0.75)
                IQR = Q3 - Q1
                df_proc[col] = df_proc[col].clip(Q1-1.5*IQR, Q3+1.5*IQR)
            df_proc = pd.get_dummies(df_proc, columns=['cp','restecg','slope','thal','ca'], drop_first=False)
            heart_cols = assets['heart_cols']
            for c in heart_cols:
                if c not in df_proc.columns:
                    df_proc[c] = 0
            X = clean_model_input(assets['scaler_h'].transform(df_proc[heart_cols]))
            y = df_proc['target'].values
            _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            mods = {'Logistic Regression': assets['lr_h'],
                    'KNN': assets['knn_h'],
                    'Decision Tree': assets['dt_h']}
            best_name = assets['best']['heart']
            class_labels = ['No Disease', 'Heart Disease']

        # Summary table
        rows = []
        for mname, model in mods.items():
            X_te = clean_model_input(X_te)
            y_pred = model.predict(X_te)
            rows.append({
                'Model': mname + (' ⭐' if mname == best_name else ''),
                'Accuracy':  round(accuracy_score(y_te, y_pred), 4),
                'Precision': round(precision_score(y_te, y_pred, zero_division=0), 4),
                'Recall':    round(recall_score(y_te, y_pred, zero_division=0), 4),
                'F1-Score':  round(f1_score(y_te, y_pred, zero_division=0), 4),
            })
        st.dataframe(pd.DataFrame(rows).set_index('Model'), width="stretch")

        # Confusion matrices
        st.markdown("#### Confusion Matrices")
        cm_cols = st.columns(3)
        for i, (mname, model) in enumerate(mods.items()):
            X_te = clean_model_input(X_te)
            y_pred = model.predict(X_te)
            cm = confusion_matrix(y_te, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_labels, yticklabels=class_labels,
                        ax=ax, linewidths=1)
            ax.set_title(mname + (' ⭐' if mname == best_name else ''), fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            with cm_cols[i]:
                st.pyplot(fig)
            plt.close()

        # ROC Curves
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        clrs = ['#4CAF50','#2196F3','#FF9800']
        for (mname, model), col in zip(mods.items(), clrs):
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, y_prob)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=col, lw=2, label=f'{mname} (AUC={roc_auc:.3f})')
        ax.plot([0,1],[0,1],'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves — {metrics_ds}', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature Importance
        st.markdown("#### Feature Importance")
        fi_col1, fi_col2 = st.columns(2)

        if metrics_ds == "Diabetes":
            feat_names = list(df_d.drop('Outcome', axis=1).columns)
        else:
            feat_names = assets['heart_cols']

        dt_model = mods['Decision Tree']
        lr_model = mods['Logistic Regression']

        with fi_col1:
            imp = pd.Series(dt_model.feature_importances_, index=feat_names).sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            imp.plot(kind='barh', ax=ax, color='#FF9800')
            ax.set_title('Decision Tree — Feature Importance', fontweight='bold')
            ax.set_xlabel('Importance')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with fi_col2:
            coef = pd.Series(np.abs(lr_model.coef_[0]), index=feat_names).sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            coef.plot(kind='barh', ax=ax, color='#5C85D6')
            ax.set_title('Logistic Regression — |Coefficients|', fontweight='bold')
            ax.set_xlabel('|Coefficient|')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not assets['loaded']:
        st.error("⚠️ Run the notebook first!")
    else:
        st.markdown('<div class="section-head">🔵 PCA — Dimensionality Reduction</div>', unsafe_allow_html=True)
        pca_ds = st.radio("Dataset", ["Diabetes", "Heart Disease"], horizontal=True, key='pca_ds')

        if pca_ds == "Diabetes":
            pca_model = assets['pca_d']
            scaler    = assets['scaler_d']
            df_proc   = df_d.copy()
            zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
            df_proc[zero_cols] = df_proc[zero_cols].replace(0, np.nan)
            for col in zero_cols:
                df_proc[col].fillna(df_proc[col].median(), inplace=True)
            X_full = clean_model_input(scaler.transform(df_proc.drop('Outcome', axis=1)))
            y_full = df_proc['Outcome'].values
            label_map = {0: 'No Diabetes', 1: 'Diabetes'}
            colors_pca = {0: '#4CAF50', 1: '#F44336'}
        else:
            # Heart Disease
            pca_model = assets['pca_h']
            scaler    = assets['scaler_h']
            df_proc   = df_h.copy()
            df_enc    = pd.get_dummies(df_proc, columns=['cp','restecg','slope','thal','ca'], drop_first=False)
            hcols     = assets['heart_cols']
            for c in hcols:
                if c not in df_enc.columns:
                    df_enc[c] = 0
            X_full = clean_model_input(scaler.transform(df_enc[hcols]))
            y_full = df_enc['target'].values
            label_map  = {0: 'No Disease', 1: 'Heart Disease'}
            colors_pca = {0: '#2196F3', 1: '#FF5722'}

        # Explained variance
        pca_full = type(pca_model)(n_components=None, random_state=42)
        pca_full.fit(X_full)
        exp_var = pca_full.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(range(1, len(exp_var)+1), exp_var * 100, color='#5C85D6', alpha=0.8, label='Individual')
            ax.plot(range(1, len(cum_var)+1), cum_var * 100, 'r-o', lw=2, ms=4, label='Cumulative')
            ax.axhline(95, color='green', ls='--', lw=1.5, label='95% threshold')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance (%)')
            ax.set_title(f'{pca_ds} — PCA Explained Variance', fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            n_95 = np.argmax(cum_var >= 0.95) + 1
            st.info(f"✅ **{n_95} components** explain 95% of variance (out of {X_full.shape[1]} features)")

        with col2:
            pca2 = type(pca_model)(n_components=2, random_state=42)
            X_2d = pca2.fit_transform(X_full)
            fig, ax = plt.subplots(figsize=(7, 5))
            for cls, lbl in label_map.items():
                mask = y_full == cls
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=colors_pca[cls], label=lbl, alpha=0.6, s=25)
            ax.set_xlabel(f'PC1 ({exp_var[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({exp_var[1]*100:.1f}%)')
            ax.set_title(f'{pca_ds} — PCA 2D Scatter', fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#888;font-size:0.85rem;'>"
    "Healthcare Risk Prediction System | CSL-487 Data Science Lab | Bahria University | Spring 2026"
    "</center>",
    unsafe_allow_html=True
)

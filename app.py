import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide", page_title="Flood Records Analysis", page_icon="🌊")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

CONFIG = {
    "file_path": "FloodData.xlsx", 
    "barangay_col": "Barangay",
    "cause_col": "Flood Cause",
    "water_col": "Water Level",
    "families_col": "No. of Families affected",
    "infra_col": "Damage Infrastructure",
    "agri_col": "Damage Agriculture",
    "month_col": "Month",
    "year_col": "Year",
    "day_col": "Day",
    "sheet_index": 0,
    "severity_low": {"water": 2.0, "families": 50, "damage": 100000},
    "severity_mod": {"water": 4.0, "families": 200, "damage": 500000},
    "kmeans_k_range": range(2, 8),
    "apriori_min_support": 0.05,
    "apriori_min_confidence": 0.4,
    "apriori_min_lift": 1.0,
    "train_test_split": 0.3,
    "random_state": 42,
}

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

def clean_text(v):
    if pd.isna(v): return v
    s = str(v)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_number(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = clean_text(v).replace("ft.", "").replace("ft", "").replace("feet", "").replace(",", "").strip()
    if s == "": return np.nan
    try: return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            try: return float(m.group(0))
            except ValueError: return np.nan
        return np.nan

def parse_currency(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = clean_text(v).replace("php", "").replace("₱", "").replace(",", "").strip()
    if s == "": return np.nan
    try: return float(s)
    except ValueError: return np.nan

def parse_month(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, np.integer)): return int(v) if 1 <= int(v) <= 12 else np.nan
    if isinstance(v, float):
        if 1 <= v <= 12 and float(v).is_integer(): return int(v)
        return np.nan
    s = clean_text(v).lower()
    if s in MONTH_MAP: return MONTH_MAP[s]
    try:
        n = int(float(s))
        return n if 1 <= n <= 12 else np.nan
    except ValueError: return np.nan

def parse_year(v):
    if pd.isna(v): return np.nan
    s = clean_text(str(v))
    try: return int(float(s))
    except ValueError:
        m = re.search(r"(19|20)\d{2}", s)
        return int(m.group(0)) if m else np.nan

def normalize_barangay(v):
    if pd.isna(v): return np.nan
    return clean_text(v).title()

def normalize_cause(v):
    if pd.isna(v): return np.nan
    return clean_text(v).replace("Tropica Storm", "Tropical Storm")

def severity_label(row, cfg):
    w, f, d = row["water_level_num"], row["families_num"], row["damage_total_num"]
    lows, mods = cfg["severity_low"], cfg["severity_mod"]
    high_count = 0
    if pd.notna(w) and w > mods["water"]: high_count += 1
    if pd.notna(f) and f > mods["families"]: high_count += 1
    if pd.notna(d) and d > mods["damage"]: high_count += 1
    if high_count >= 2: return "High"
    low_all = ((pd.isna(w) or w <= lows["water"]) and (pd.isna(f) or f < lows["families"]) and (pd.isna(d) or d < lows["damage"]))
    return "Low" if low_all else "Moderate"

@st.cache_data(show_spinner=False)
def load_and_clean(file_path):
    df = pd.read_excel(file_path, sheet_name=CONFIG["sheet_index"])
    df.columns = [clean_text(c) for c in df.columns]
    df = df.dropna(how="all")
    for c in df.columns: df[c] = df[c].apply(clean_text)
    
    bc, cc, wc, fc, ic, ac, mc, yc, dc = CONFIG["barangay_col"], CONFIG["cause_col"], CONFIG["water_col"], CONFIG["families_col"], CONFIG["infra_col"], CONFIG["agri_col"], CONFIG["month_col"], CONFIG["year_col"], CONFIG["day_col"]
    
    df[bc] = df[bc].apply(normalize_barangay)
    df[cc] = df[cc].apply(normalize_cause)
    df["water_level_num"] = df[wc].apply(parse_number)
    df["families_num"] = df[fc].apply(parse_number)
    df["damage_infra_num"] = df[ic].apply(parse_currency)
    df["damage_agri_num"] = df[ac].apply(parse_currency)
    df["damage_total_num"] = df[["damage_infra_num", "damage_agri_num"]].sum(axis=1, min_count=1)
    df["month_num"] = df[mc].apply(parse_month)
    df["year_num"] = df[yc].apply(parse_year)
    df["day_num"] = df[dc].apply(parse_number)
    
    df = df.dropna(subset=[bc, cc])
    df["water_level_for_sev"] = df["water_level_num"].where(df["water_level_num"] <= 20)
    
    df["severity"] = df.apply(lambda r: severity_label(
        pd.Series({"water_level_num": r["water_level_for_sev"], "families_num": r["families_num"], "damage_total_num": r["damage_total_num"]}), CONFIG), axis=1)
        
    df["year_month"] = df["year_num"].astype("Int64").astype(str) + "-" + df["month_num"].astype("Int64").astype(str).str.zfill(2)
    return df

try:
    df = load_and_clean(CONFIG["file_path"])
except FileNotFoundError:
    st.error(f"Data file not found. Please ensure '{CONFIG['file_path']}' is uploaded to the same folder as app.py.")
    st.stop()

with st.sidebar:
    st.title("🌊 Flood Dashboard")
    st.markdown("Automated Data Mining Analysis")
    st.divider()
    
    total = len(df)
    st.metric("Total Flood Records", f"{total:,}")
    st.metric("Affected Barangays", df[CONFIG["barangay_col"]].nunique())
    st.metric("Families Affected", f"{int(df['families_num'].sum(skipna=True)):,}")
    st.metric("Total Damage", f"₱{int(df['damage_total_num'].sum(skipna=True)):,}")

tabs = st.tabs(["Overview", "Seasonality & Weather", "Flood Risk Clusters", "Severity Classification", "Association Rules", "Decision Support"])
SEV_COLORS = {"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728"}

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Records by Barangay")
        bc = df[CONFIG["barangay_col"]].value_counts().reset_index()
        bc.columns = ["Barangay", "Records"]
        st.plotly_chart(px.bar(bc, x="Barangay", y="Records", color="Records", color_continuous_scale="Blues", template="plotly_white"), use_container_width=True)
    with col2:
        st.subheader("Severity Distribution")
        sev = df["severity"].value_counts().reindex(["Low", "Moderate", "High"]).fillna(0).astype(int).reset_index()
        sev.columns = ["Severity", "Records"]
        st.plotly_chart(px.pie(sev, names="Severity", values="Records", color="Severity", color_discrete_map=SEV_COLORS, hole=0.4), use_container_width=True)
        st.info(f"High-severity events account for **{(sev[sev['Severity']=='High']['Records'].sum() / max(total, 1)) * 100:.1f}%** of all records.")

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Records by Month")
        mcount = df["month_num"].dropna().astype(int).value_counts().sort_index().reset_index()
        mcount.columns = ["Month_Num", "Records"]
        mcount["Month"] = mcount["Month_Num"].apply(lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
        st.plotly_chart(px.bar(mcount, x="Month", y="Records", color_discrete_sequence=["#17becf"], template="plotly_white"), use_container_width=True)
    with col2:
        st.subheader("Weather System Frequency")
        wcount = df[CONFIG["cause_col"]].value_counts().reset_index()
        wcount.columns = ["Weather System", "Records"]
        st.plotly_chart(px.bar(wcount, x="Weather System", y="Records", color_discrete_sequence=["#9467bd"], template="plotly_white"), use_container_width=True)

    st.subheader("Records by Year")
    ycount = df["year_num"].dropna().astype(int).value_counts().sort_index().reset_index()
    ycount.columns = ["Year", "Records"]
    st.plotly_chart(px.line(ycount, x="Year", y="Records", markers=True, template="plotly_white"), use_container_width=True)

with tabs[2]:
    st.subheader("K-Means Clustering of Flood Events")
    feat = df[["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]].copy().fillna(df[["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]].median(numeric_only=True))
    feat["barangay_code"] = LabelEncoder().fit_transform(df[CONFIG["barangay_col"]].astype(str))
    feat["cause_code"] = LabelEncoder().fit_transform(df[CONFIG["cause_col"]].astype(str))
    Xs = StandardScaler().fit_transform(feat.values)
    k_range = list(CONFIG["kmeans_k_range"])
    inertias, sil_scores = [], []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(Xs, labels) if k > 1 else np.nan)
            
    best_k = k_range[int(np.nanargmax(sil_scores))]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success(f"Optimal Clusters (k): **{best_k}**\n\nSilhouette Score: **{sil_scores[int(np.nanargmax(sil_scores))]:.3f}**")
        st.dataframe(pd.DataFrame({"k": k_range, "WSS": inertias, "Silhouette": sil_scores}).round(3), hide_index=True)
    with col2:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=k_range, y=inertias, mode='lines+markers', name='WSS', yaxis='y1'))
        fig_elbow.add_trace(go.Scatter(x=k_range, y=sil_scores, mode='lines+markers', name='Silhouette', yaxis='y2'))
        fig_elbow.update_layout(yaxis=dict(title="WSS"), yaxis2=dict(title="Silhouette", overlaying='y', side='right'), template="plotly_white", margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_elbow, use_container_width=True)

    df["cluster"] = KMeans(n_clusters=best_k, random_state=CONFIG["random_state"], n_init=10).fit_predict(Xs)
    st.dataframe(df.groupby("cluster").agg(Events=("cluster", "size"), Avg_Water_Level=("water_level_num", "mean"), Avg_Families=("families_num", "mean"), Avg_Infra_Damage=("damage_infra_num", "mean"), Avg_Agri_Damage=("damage_agri_num", "mean")).round(2), use_container_width=True)
    st.plotly_chart(px.imshow(pd.crosstab(df["cluster"], df[CONFIG["barangay_col"]]), text_auto=True, aspect="auto", color_continuous_scale="Blues"), use_container_width=True)

with tabs[3]:
    st.subheader("Decision Tree Severity Classifier")
    model_cols = ["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]
    mdf = df.dropna(subset=["severity"]).copy()
    
    Xm = mdf[model_cols].fillna(mdf[model_cols].median()).to_numpy(dtype=float)
    ym = mdf["severity"].to_numpy()
    
    min_class_count = pd.Series(ym).value_counts().min()
    stratify_arg = ym if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(Xm, ym, test_size=CONFIG["train_test_split"], random_state=CONFIG["random_state"], stratify=stratify_arg)
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=CONFIG["random_state"]).fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    
    col1, col2 = st.columns(2)
    with col1:
        labels = sorted(set(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        st.write("**Confusion Matrix**")
        st.plotly_chart(px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Oranges", labels=dict(x="Predicted", y="Actual")), use_container_width=True)
    with col2:
        st.write("**Feature Importance**")
        imp = pd.Series(dt.feature_importances_, index=model_cols).sort_values(ascending=True).reset_index()
        imp.columns = ["Feature", "Importance"]
        st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation='h', color_discrete_sequence=["#ff7f0e"]), use_container_width=True)

with tabs[4]:
    st.subheader("Apriori Association Rules")
    
    try:
        adf = df.copy().dropna(subset=["severity"]).reset_index(drop=True)
        
        trans = pd.DataFrame(index=adf.index)
        trans["Barangay_" + adf[CONFIG["barangay_col"]].astype(str)] = True
        trans["Cause_" + adf[CONFIG["cause_col"]].astype(str)] = True
        trans["Severity_" + adf["severity"].astype(str)] = True
        
        # Fill missing months with 'Unknown' to prevent index misalignment
        month_names = adf["month_num"].apply(lambda m: pd.Timestamp(2024, int(m), 1).strftime("%B") if pd.notna(m) else "Unknown")
        trans["Month_" + month_names] = True
        
        trans = trans.fillna(False).astype(bool)
        
        freq = apriori(trans, min_support=CONFIG["apriori_min_support"], use_colnames=True)
        
        if freq.empty:
            st.warning("No itemsets met the minimum support threshold.")
        else:
            rules = association_rules(freq, metric="confidence", min_threshold=CONFIG["apriori_min_confidence"])
            rules = rules[rules["lift"] >= CONFIG["apriori_min_lift"]]
            
            if rules.empty:
                st.warning("No rules met the confidence and lift thresholds.")
            else:
                rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
                rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
                st.dataframe(rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].sort_values("lift", ascending=False).round(3), use_container_width=True, hide_index=True)
                st.plotly_chart(px.scatter(rules, x="support", y="confidence", color="lift", hover_data=["antecedents_str", "consequents_str"], color_continuous_scale="Viridis", template="plotly_white"), use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate rules: {e}")

with tabs[5]:
    st.subheader("Derived Decision Support Priorities")
    col1, col2 = st.columns(2)
    with col1:
        bc = df[CONFIG["barangay_col"]].value_counts().head(10).reset_index()
        bc.columns = ["Barangay", "Records"]
        fig_b = px.bar(bc, x="Records", y="Barangay", orientation='h', color_discrete_sequence=["#1f77b4"])
        fig_b.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
        st.write("**Top Barangays by Flood Records**")
        st.plotly_chart(fig_b, use_container_width=True)
    with col2:
        high_by_b = df[df["severity"] == "High"][CONFIG["barangay_col"]].value_counts().head(10).reset_index()
        high_by_b.columns = ["Barangay", "High-Severity Records"]
        fig_hb = px.bar(high_by_b, x="High-Severity Records", y="Barangay", orientation='h', color_discrete_sequence=["#d62728"])
        fig_hb.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
        st.write("**Top Barangays by High-Severity Events**")
        st.plotly_chart(fig_hb, use_container_width=True)

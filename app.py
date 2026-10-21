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

# Custom CSS for a modern look
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
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
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1f77b4;
    }
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
    if pd.isna(v):
        return v
    s = str(v)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_number(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = clean_text(v)
    s = s.replace("ft.", "").replace("ft", "").replace("feet", "")
    s = s.replace(",", "").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return np.nan
        return np.nan

def parse_currency(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = clean_text(v).replace("php", "").replace("₱", "").strip()
    s = s.replace(",", "")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def parse_month(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return int(v) if 1 <= int(v) <= 12 else np.nan
    if isinstance(v, float):
        if 1 <= v <= 12 and float(v).is_integer():
            return int(v)
        return np.nan
    s = clean_text(v).lower()
    if s in MONTH_MAP:
        return MONTH_MAP[s]
    try:
        n = int(float(s))
        return n if 1 <= n <= 12 else np.nan
    except ValueError:
        return np.nan

def parse_year(v):
    if pd.isna(v):
        return np.nan
    s = clean_text(str(v))
    try:
        return int(float(s))
    except ValueError:
        m = re.search(r"(19|20)\d{2}", s)
        return int(m.group(0)) if m else np.nan

def normalize_barangay(v):
    if pd.isna(v):
        return np.nan
    s = clean_text(v).title()
    return s

def normalize_cause(v):
    if pd.isna(v):
        return np.nan
    s = clean_text(v)
    s = s.replace("Tropica Storm", "Tropical Storm")
    return s

def severity_label(row, cfg):
    w = row["water_level_num"]
    f = row["families_num"]
    d = row["damage_total_num"]
    lows = cfg["severity_low"]
    mods = cfg["severity_mod"]
    high_count = 0
    if pd.notna(w) and w > mods["water"]:
        high_count += 1
    if pd.notna(f) and f > mods["families"]:
        high_count += 1
    if pd.notna(d) and d > mods["damage"]:
        high_count += 1
    if high_count >= 2:
        return "High"
    low_all = (
        (pd.isna(w) or w <= lows["water"]) and
        (pd.isna(f) or f < lows["families"]) and
        (pd.isna(d) or d < lows["damage"])
    )
    if low_all:
        return "Low"
    return "Moderate"

@st.cache_data(show_spinner=False)
def load_and_clean(file_path):
    df = pd.read_excel(file_path, sheet_name=CONFIG["sheet_index"])
    df.columns = [clean_text(c) for c in df.columns]
    df = df.dropna(how="all")
    for c in df.columns:
        df[c] = df[c].apply(clean_text)
    bc = CONFIG["barangay_col"]
    cc = CONFIG["cause_col"]
    wc = CONFIG["water_col"]
    fc = CONFIG["families_col"]
    ic = CONFIG["infra_col"]
    ac = CONFIG["agri_col"]
    mc = CONFIG["month_col"]
    yc = CONFIG["year_col"]
    dc = CONFIG["day_col"]
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
        pd.Series({
            "water_level_num": r["water_level_for_sev"],
            "families_num": r["families_num"],
            "damage_total_num": r["damage_total_num"],
        }), CONFIG), axis=1)
    df["year_month"] = df["year_num"].astype("Int64").astype(str) + "-" + df["month_num"].astype("Int64").astype(str).str.zfill(2)
    return df

try:
    df = load_and_clean(CONFIG["file_path"])
except FileNotFoundError:
    st.error(f"Data file not found. Please ensure '{CONFIG['file_path']}' is uploaded to the same folder as app.py.")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🌊 Flood Dashboard")
    st.markdown("Automated Data Mining Analysis")
    st.divider()
    
    total = len(df)
    st.metric("Total Flood Records", f"{total:,}")
    st.metric("Affected Barangays", df[CONFIG["barangay_col"]].nunique())
    st.metric("Families Affected", f"{int(df['families_num'].sum(skipna=True)):,}")
    st.metric("Total Damage", f"₱{int(df['damage_total_num'].sum(skipna=True)):,}")

# Main Area Tabs
tabs = st.tabs([
    "Overview",
    "Seasonality & Weather",
    "Flood Risk Clusters",
    "Severity Classification",
    "Association Rules",
    "Decision Support",
])

# Color Palette
SEV_COLORS = {"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728"}

with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Records by Barangay")
        bc = df[CONFIG["barangay_col"]].value_counts().reset_index()
        bc.columns = ["Barangay", "Records"]
        fig_bar = px.bar(bc, x="Barangay", y="Records", color="Records", 
                         color_continuous_scale="Blues", template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Severity Distribution")
        sev = df["severity"].value_counts().reindex(["Low", "Moderate", "High"]).fillna(0).astype(int).reset_index()
        sev.columns = ["Severity", "Records"]
        fig_pie = px.pie(sev, names="Severity", values="Records", 
                         color="Severity", color_discrete_map=SEV_COLORS, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        high_pct = (sev[sev["Severity"]=="High"]["Records"].sum() / max(total, 1)) * 100
        st.info(f"High-severity events account for **{high_pct:.1f}%** of all records.")

with tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Records by Month")
        mcount = df["month_num"].dropna().astype(int).value_counts().sort_index().reset_index()
        mcount.columns = ["Month_Num", "Records"]
        mcount["Month"] = mcount["Month_Num"].apply(lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
        fig_month = px.bar(mcount, x="Month", y="Records", color_discrete_sequence=["#17becf"], template="plotly_white")
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        st.subheader("Weather System Frequency")
        wcount = df[CONFIG["cause_col"]].value_counts().reset_index()
        wcount.columns = ["Weather System", "Records"]
        fig_weather = px.bar(wcount, x="Weather System", y="Records", color_discrete_sequence=["#9467bd"], template="plotly_white")
        st.plotly_chart(fig_weather, use_container_width=True)

    st.subheader("Records by Year")
    ycount = df["year_num"].dropna().astype(int).value_counts().sort_index().reset_index()
    ycount.columns = ["Year", "Records"]
    fig_year = px.line(ycount, x="Year", y="Records", markers=True, template="plotly_white")
    st.plotly_chart(fig_year, use_container_width=True)

    with st.expander("View Detailed Crosstab Data"):
        st.write("**Weather System by Severity**")
        ct = pd.crosstab(df[CONFIG["cause_col"]], df["severity"])
        st.dataframe(ct, use_container_width=True)

with tabs[2]:
    st.subheader("K-Means Clustering of Flood Events")
    num_cols = ["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]
    feat = df[num_cols].copy()
    feat = feat.fillna(feat.median(numeric_only=True))
    le_b = LabelEncoder()
    le_c = LabelEncoder()
    feat["barangay_code"] = le_b.fit_transform(df[CONFIG["barangay_col"]].astype(str))
    feat["cause_code"] = le_c.fit_transform(df[CONFIG["cause_col"]].astype(str))
    X = feat.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k_range = list(CONFIG["kmeans_k_range"])
    inertias = []
    sil_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        if k > 1:
            sil_scores.append(silhouette_score(Xs, labels))
        else:
            sil_scores.append(np.nan)
            
    best_idx = int(np.nanargmax(sil_scores))
    best_k = k_range[best_idx]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success(f"Optimal Clusters (k): **{best_k}**\n\nSilhouette Score: **{sil_scores[best_idx]:.3f}**")
        elbow_df = pd.DataFrame({"k": k_range, "WSS": inertias, "Silhouette": sil_scores})
        st.dataframe(elbow_df.round(3), use_container_width=True, hide_index=True)
        
    with col2:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=k_range, y=inertias, mode='lines+markers', name='WSS', yaxis='y1'))
        fig_elbow.add_trace(go.Scatter(x=k_range, y=sil_scores, mode='lines+markers', name='Silhouette', yaxis='y2'))
        fig_elbow.update_layout(
            title="Elbow Method & Silhouette Score",
            yaxis=dict(title="WSS"),
            yaxis2=dict(title="Silhouette", overlaying='y', side='right'),
            template="plotly_white"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    km = KMeans(n_clusters=best_k, random_state=CONFIG["random_state"], n_init=10)
    df["cluster"] = km.fit_predict(Xs)
    
    st.subheader("Cluster Profiles")
    prof = df.groupby("cluster").agg(
        Events=("cluster", "size"),
        Avg_Water_Level=("water_level_num", "mean"),
        Avg_Families=("families_num", "mean"),
        Avg_Infra_Damage=("damage_infra_num", "mean"),
        Avg_Agri_Damage=("damage_agri_num", "mean"),
    ).round(2)
    st.dataframe(prof, use_container_width=True)
    
    st.subheader("Barangay Distribution Across Clusters")
    ct3 = pd.crosstab(df["cluster"], df[CONFIG["barangay_col"]])
    fig_heat = px.imshow(ct3, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_heat, use_container_width=True)

with tabs[3]:
    st.subheader("Decision Tree Severity Classifier")
    model_cols = ["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]
    mdf = df.dropna(subset=["severity"]).copy()
    Xm = mdf[model_cols].fillna(mdf[model_cols].median()).values
    ym = mdf["severity"].values
    
    min_class_count = pd.Series(ym).value_counts().min()
    stratify_arg = ym if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        Xm, ym, test_size=CONFIG["train_test_split"], random_state=CONFIG["random_state"], stratify=stratify_arg
    )
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=CONFIG["random_state"])
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Confusion Matrix**")
        labels = sorted(set(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Oranges", labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        st.write("**Feature Importance**")
        imp = pd.Series(dt.feature_importances_, index=model_cols).sort_values(ascending=True).reset_index()
        imp.columns = ["Feature", "Importance"]
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation='h', color_discrete_sequence=["#ff7f0e"])
        st.plotly_chart(fig_imp, use_container_width=True)

with tabs[4]:
    st.subheader("Apriori Association Rules")
    st.write("Mining rules based on Weather Cause, Barangay, Month, and Severity.")
    
    adf = df.copy()
    adf = adf.dropna(subset=["severity"])
    trans = pd.DataFrame(index=adf.index)
    trans["Barangay_" + adf[CONFIG["barangay_col"]].astype(str)] = True
    trans["Cause_" + adf[CONFIG["cause_col"]].astype(str)] = True
    trans["Severity_" + adf["severity"].astype(str)] = True
    month_names = adf["month_num"].dropna().astype(int).apply(lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
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
            show = rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].sort_values("lift", ascending=False)
            st.dataframe(show.round(3), use_container_width=True, hide_index=True)
            
            fig_rules = px.scatter(rules, x="support", y="confidence", color="lift", 
                                   hover_data=["antecedents_str", "consequents_str"],
                                   color_continuous_scale="Viridis", template="plotly_white")
            st.plotly_chart(fig_rules, use_container_width=True)

with tabs[5]:
    st.subheader("Derived Decision Support Priorities")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top Barangays by Flood Records**")
        bc = df[CONFIG["barangay_col"]].value_counts().head(10).reset_index()
        bc.columns = ["Barangay", "Records"]
        fig_b = px.bar(bc, x="Records", y="Barangay", orientation='h', color_discrete_sequence=["#1f77b4"])
        fig_b.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_b, use_container_width=True)
        
    with col2:
        st.write("**Top Barangays by High-Severity Events**")
        high_by_b = df[df["severity"] == "High"][CONFIG["barangay_col"]].value_counts().head(10).reset_index()
        high_by_b.columns = ["Barangay", "High-Severity Records"]
        fig_hb = px.bar(high_by_b, x="High-Severity Records", y="Barangay", orientation='h', color_discrete_sequence=["#d62728"])
        fig_hb.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_hb, use_container_width=True)
        
    st.write("**Primary Flood Drivers (Share of Total Records)**")
    cause_share = df[CONFIG["cause_col"]].value_counts(normalize=True).mul(100).round(1).reset_index()
    cause_share.columns = ["Weather System", "Percentage (%)"]
    fig_share = px.treemap(cause_share, path=["Weather System"], values="Percentage (%)", color="Percentage (%)", color_continuous_scale="Blues")
    st.plotly_chart(fig_share, use_container_width=True)
    
    min_year, max_year = int(df['year_num'].min()), int(df['year_num'].max())
    st.info(f"Data records span from **{min_year}** to **{max_year}**.")

import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score

st.set_page_config(layout="wide", page_title="Flood Analysis Dashboard", page_icon="🌊")

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
    "sheet_index": 0,
    "severity_low": {"water": 2.0, "families": 50, "damage": 100000},
    "severity_mod": {"water": 4.0, "families": 200, "damage": 500000},
    "train_test_split": 0.3,
    "random_state": 42,
}

MONTH_MAP = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}

def clean_text(v):
    if pd.isna(v): return v
    s = unicodedata.normalize("NFKC", str(v))
    s = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def parse_number(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = clean_text(v).replace("ft.", "").replace("ft", "").replace("feet", "").replace(",", "").strip()
    if not s: return np.nan
    try: return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else np.nan

def parse_currency(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = clean_text(v).replace("php", "").replace("₱", "").replace(",", "").strip()
    try: return float(s)
    except: return np.nan

def parse_month(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, np.integer)): return int(v) if 1 <= int(v) <= 12 else np.nan
    s = clean_text(v).lower()
    if s in MONTH_MAP: return MONTH_MAP[s]
    try:
        n = int(float(s))
        return n if 1 <= n <= 12 else np.nan
    except: return np.nan

def parse_year(v):
    if pd.isna(v): return np.nan
    try: return int(float(str(v)))
    except:
        m = re.search(r"(19|20)\d{2}", str(v))
        return int(m.group(0)) if m else np.nan

def severity_label(row):
    w, f, d = row["water_level_num"], row["families_num"], row["damage_total_num"]
    high_count = 0
    if pd.notna(w) and w > 4.0: high_count += 1
    if pd.notna(f) and f > 200: high_count += 1
    if pd.notna(d) and d > 500000: high_count += 1
    if high_count >= 2: return "High"
    if (pd.isna(w) or w <= 2.0) and (pd.isna(f) or f < 50) and (pd.isna(d) or d < 100000): return "Low"
    return "Moderate"

def simple_apriori(transactions, min_support=0.05, min_confidence=0.4):
    n = len(transactions)
    items = set(item for t in transactions for item in t)
    support_counts = {frozenset([i]): sum(1 for t in transactions if i in t) for i in items}
    L1 = {k: v/n for k, v in support_counts.items() if v/n >= min_support}
    
    rules = []
    for k in range(2, 4):
        itemsets = list(L1.keys())
        candidates = [frozenset(c) for c in combinations(items, k)]
        valid = {}
        for c in candidates:
            count = sum(1 for t in transactions if c.issubset(t))
            sup = count / n
            if sup >= min_support: valid[c] = sup
        
        for c, sup in valid.items():
            for i in c:
                ant = c - frozenset([i])
                if ant in L1:
                    conf = sup / L1[ant]
                    if conf >= min_confidence:
                        rules.append((ant, frozenset([i]), sup, conf, conf/L1[frozenset([i])]))
        L1.update(valid)
    return rules

@st.cache_data(show_spinner=False)
def load_and_clean(file_path):
    df = pd.read_excel(file_path, sheet_name=CONFIG["sheet_index"])
    df.columns = [clean_text(c) for c in df.columns]
    df = df.dropna(how="all")
    for c in df.columns: df[c] = df[c].apply(clean_text)
    
    bc, cc, wc, fc, ic, ac = CONFIG["barangay_col"], CONFIG["cause_col"], CONFIG["water_col"], CONFIG["families_col"], CONFIG["infra_col"], CONFIG["agri_col"]
    
    df[bc] = df[bc].apply(lambda x: clean_text(x).title() if pd.notna(x) else np.nan)
    df[cc] = df[cc].apply(lambda x: clean_text(x).replace("Tropica Storm", "Tropical Storm") if pd.notna(x) else np.nan)
    
    df["water_level_num"] = df[wc].apply(parse_number)
    df["families_num"] = df[fc].apply(parse_number)
    df["damage_infra_num"] = df[ic].apply(parse_currency)
    df["damage_agri_num"] = df[ac].apply(parse_currency)
    df["damage_total_num"] = df[["damage_infra_num", "damage_agri_num"]].sum(axis=1, min_count=1)
    df["month_num"] = df[CONFIG["month_col"]].apply(parse_month)
    df["year_num"] = df[CONFIG["year_col"]].apply(parse_year)
    
    df = df.dropna(subset=[bc, cc])
    df["severity"] = df.apply(severity_label, axis=1)
    return df

try:
    df = load_and_clean(CONFIG["file_path"])
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.title("🌊 Flood Dashboard")
st.sidebar.metric("Total Flood Records", f"{len(df):,}")
st.sidebar.metric("Affected Barangays", df[CONFIG["barangay_col"]].nunique())
st.sidebar.metric("Families Affected", f"{int(df['families_num'].sum(skipna=True)):,}")
st.sidebar.metric("Total Damage (PHP)", f"₱{int(df['damage_total_num'].sum(skipna=True)):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Seasonality", "Clustering", "Decision Tree", "Rules & Support"])

with tab1:
    st.header("Flood Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Records by Barangay")
        bc = df[CONFIG["barangay_col"]].value_counts()
        st.bar_chart(bc)
    with col2:
        st.subheader("Severity Distribution")
        sev = df["severity"].value_counts().reindex(["Low", "Moderate", "High"]).fillna(0)
        st.bar_chart(sev)

with tab2:
    st.header("Seasonality & Weather")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flood Frequency by Month")
        mcount = df["month_num"].dropna().astype(int).value_counts().sort_index()
        mcount.index = [pd.Timestamp(2024, m, 1).strftime("%B") for m in mcount.index]
        st.bar_chart(mcount)
    with col2:
        st.subheader("Weather System Frequency")
        wcount = df[CONFIG["cause_col"]].value_counts()
        st.bar_chart(wcount)

with tab3:
    st.header("K-Means Clustering (Flood Profiles)")
    st.write("Groups similar flood events to identify distinct risk profiles.")
    try:
        feat = df[["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]].fillna(0)
        Xs = StandardScaler().fit_transform(feat)
        
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
        df["cluster"] = km.labels_
        
        st.success("Clustering complete. The data was grouped into 3 distinct risk profiles.")
        prof = df.groupby("cluster").agg(
            Events=("cluster", "size"),
            Avg_Water_Level=("water_level_num", "mean"),
            Avg_Families=("families_num", "mean"),
            Avg_Damage=("damage_total_num", "mean")
        ).round(2)
        st.dataframe(prof)
        st.bar_chart(df["cluster"].value_counts())
    except Exception as e:
        st.error(f"Clustering error: {e}")

with tab4:
    st.header("Decision Tree Severity Classification")
    st.write("Trains a model to predict severity based on damage and water levels.")
    try:
        model_cols = ["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]
        Xm = df[model_cols].fillna(0).values
        ym = df["severity"].values
        
        X_train, X_test, y_train, y_test = train_test_split(Xm, ym, test_size=0.3, random_state=42)
        dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        
        st.subheader("Feature Importance (What causes High Severity?)")
        imp = pd.Series(dt.feature_importances_, index=model_cols).sort_values()
        st.bar_chart(imp)
    except Exception as e:
        st.error(f"Decision Tree error: {e}")

with tab5:
    st.header("Association Rules & Decision Support")
    st.write("Generates 'If-Then' rules based on historical data to guide preparedness.")
    
    try:
        adf = df.dropna(subset=["severity"]).copy()
        transactions = []
        for _, row in adf.iterrows():
            t = {f"Brgy_{row[CONFIG['barangay_col']]}", f"Severity_{row['severity']}"}
            if pd.notna(row["month_num"]):
                t.add(f"Month_{pd.Timestamp(2024, int(row['month_num']), 1).strftime('%B')}")
            transactions.append(t)
            
        rules = simple_apriori(transactions, min_support=0.05, min_confidence=0.4)
        rules = [r for r in rules if r[4] > 1.0]
        rules.sort(key=lambda x: x[3], reverse=True)
        
        if rules:
            rule_data = [{"If (Antecedent)": ", ".join(list(r[0])), "Then (Consequent)": ", ".join(list(r[1])), "Confidence": f"{r[3]*100:.1f}%", "Lift": round(r[4], 2)} for r in rules[:15]]
            st.dataframe(pd.DataFrame(rule_data), use_container_width=True, hide_index=True)
        else:
            st.info("No strong rules met the thresholds.")
            
        st.divider()
        st.subheader("Dynamic Recommendations (Obj 3)")
        st.success(f"""
        **Based on the Clustering and Decision Tree Results:**
        *Target Barangay:* **{df[df['severity']=='High'][CONFIG['barangay_col']].value_counts().index[0]}**
        *Action:* This area has the highest number of High-Severity floods. The LGU should prioritize budget allocation for evacuation centers and pre-position relief goods here.
        
        **Based on the Seasonality & Association Rules:**
        *Target Month:* **{pd.Timestamp(2024, int(df['month_num'].dropna().value_counts().index[0]), 1).strftime('%B')}**
        *Action:* Historical data shows floods peak this month. Mandatory drills and river dredging should be completed before this period.
        """)
    except Exception as e:
        st.error(f"Rule mining error: {e}")

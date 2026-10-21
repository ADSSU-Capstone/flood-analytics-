import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide", page_title="Flood Records Analysis")

CONFIG = {
    "file_path": "FloodData.xlsx", # Make sure this matches your exact filename in GitHub
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

# Load the data automatically
try:
    df = load_and_clean(CONFIG["file_path"])
except FileNotFoundError:
    st.error(f"Data file not found. Please ensure '{CONFIG['file_path']}' is uploaded to the same folder as app.py in your GitHub repository.")
    st.stop()

st.title("Flood Records Dashboard")

tabs = st.tabs([
    "Overview",
    "Seasonality & Weather",
    "Flood Risk Clusters",
    "Severity Classification",
    "Association Rules",
    "Decision Support",
])

with tabs[0]:
    total = len(df)
    st.metric("Total flood records", f"{total:,}")
    st.metric("Barangays affected", df[CONFIG["barangay_col"]].nunique())
    st.metric("Weather systems recorded", df[CONFIG["cause_col"]].nunique())
    st.metric("Total families affected", f"{int(df['families_num'].sum(skipna=True)):,}")
    st.metric("Total damage (PHP)", f"{int(df['damage_total_num'].sum(skipna=True)):,}")

    st.subheader("Records by barangay")
    bc = df[CONFIG["barangay_col"]].value_counts()
    st.dataframe(bc.rename_axis("Barangay").reset_index(name="Records"), use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bc.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Barangay")
    ax.set_ylabel("Records")
    st.pyplot(fig)

    st.subheader("Severity distribution")
    sev = df["severity"].value_counts().reindex(["Low", "Moderate", "High"]).fillna(0).astype(int)
    st.dataframe(sev.rename_axis("Severity").reset_index(name="Records"), use_container_width=True)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sev.plot(kind="bar", ax=ax2, color=["#4caf50", "#ff9800", "#f44336"])
    ax2.set_ylabel("Records")
    st.pyplot(fig2)
    high_pct = (sev.get("High", 0) / max(total, 1)) * 100
    st.write(f"High-severity events account for {high_pct:.1f}% of records.")

with tabs[1]:
    st.subheader("Records by month")
    mcount = df["month_num"].dropna().astype(int).value_counts().sort_index()
    mcount.index = [pd.Timestamp(2024, m, 1).strftime("%B") for m in mcount.index]
    st.dataframe(mcount.rename_axis("Month").reset_index(name="Records"), use_container_width=True)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    mcount.plot(kind="bar", ax=ax3, color="teal")
    ax3.set_ylabel("Records")
    st.pyplot(fig3)

    st.subheader("Records by year")
    ycount = df["year_num"].dropna().astype(int).value_counts().sort_index()
    st.dataframe(ycount.rename_axis("Year").reset_index(name="Records"), use_container_width=True)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ycount.plot(kind="bar", ax=ax4, color="darkorange")
    ax4.set_ylabel("Records")
    st.pyplot(fig4)

    st.subheader("Weather system frequency")
    wcount = df[CONFIG["cause_col"]].value_counts()
    st.dataframe(wcount.rename_axis("Weather System").reset_index(name="Records"), use_container_width=True)
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    wcount.plot(kind="bar", ax=ax5, color="slateblue")
    ax5.set_ylabel("Records")
    st.pyplot(fig5)

    st.subheader("Weather system by severity")
    ct = pd.crosstab(df[CONFIG["cause_col"]], df["severity"])
    st.dataframe(ct, use_container_width=True)
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    ct.plot(kind="bar", stacked=True, ax=ax6, color=["#4caf50", "#ff9800", "#f44336"])
    ax6.set_ylabel("Records")
    st.pyplot(fig6)

    st.subheader("Monthly severity composition")
    sev_month = df.dropna(subset=["month_num"]).assign(
        month_name=lambda d: d["month_num"].astype(int).apply(lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
    )
    ct2 = pd.crosstab(sev_month["month_name"], sev_month["severity"])
    month_order = [pd.Timestamp(2024, m, 1).strftime("%B") for m in range(1, 13)]
    ct2 = ct2.reindex(month_order).fillna(0).astype(int)
    st.dataframe(ct2, use_container_width=True)

with tabs[2]:
    st.subheader("K-Means clustering of flood events")
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
    st.write(f"Best k by silhouette coefficient: {best_k} (silhouette = {sil_scores[best_idx]:.3f})")
    elbow_df = pd.DataFrame({"k": k_range, "WSS": inertias, "Silhouette": sil_scores})
    st.dataframe(elbow_df, use_container_width=True)
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(12, 4))
    ax7a.plot(k_range, inertias, marker="o")
    ax7a.set_xlabel("k")
    ax7a.set_ylabel("WSS")
    ax7a.set_title("Elbow")
    ax7b.plot(k_range, sil_scores, marker="o", color="green")
    ax7b.set_xlabel("k")
    ax7b.set_ylabel("Silhouette")
    ax7b.set_title("Silhouette")
    st.pyplot(fig7)
    km = KMeans(n_clusters=best_k, random_state=CONFIG["random_state"], n_init=10)
    df["cluster"] = km.fit_predict(Xs)
    st.subheader("Cluster profiles")
    prof = df.groupby("cluster").agg(
        events=("cluster", "size"),
        water_mean=("water_level_num", "mean"),
        families_mean=("families_num", "mean"),
        infra_mean=("damage_infra_num", "mean"),
        agri_mean=("damage_agri_num", "mean"),
    ).round(2)
    st.dataframe(prof, use_container_width=True)
    st.subheader("Barangay distribution per cluster")
    ct3 = pd.crosstab(df["cluster"], df[CONFIG["barangay_col"]])
    st.dataframe(ct3, use_container_width=True)
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    sns.heatmap(ct3, annot=True, fmt="d", cmap="Blues", ax=ax8)
    st.pyplot(fig8)
    st.subheader("Weather system per cluster")
    ct4 = pd.crosstab(df["cluster"], df[CONFIG["cause_col"]])
    st.dataframe(ct4, use_container_width=True)

with tabs[3]:
    st.subheader("Decision tree severity classifier")
    model_cols = ["water_level_num", "families_num", "damage_infra_num", "damage_agri_num"]
    mdf = df.dropna(subset=["severity"]).copy()
    Xm = mdf[model_cols].fillna(mdf[model_cols].median()).values
    ym = mdf["severity"].values
    X_train, X_test, y_train, y_test = train_test_split(
        Xm, ym, test_size=CONFIG["train_test_split"], random_state=CONFIG["random_state"], stratify=ym
    )
    dt = DecisionTreeClassifier(max_depth=5, random_state=CONFIG["random_state"])
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("Precision (weighted)", f"{prec:.3f}")
    st.metric("Recall (weighted)", f"{rec:.3f}")
    st.metric("F1 (weighted)", f"{f1:.3f}")
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)
    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels, ax=ax9)
    ax9.set_xlabel("Predicted")
    ax9.set_ylabel("Actual")
    st.pyplot(fig9)
    st.subheader("Feature importance")
    imp = pd.Series(dt.feature_importances_, index=model_cols).sort_values(ascending=False)
    st.dataframe(imp.rename_axis("Feature").reset_index(name="Importance").round(4), use_container_width=True)

with tabs[4]:
    st.subheader("Apriori association rules")
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
        st.write("No itemsets met the minimum support threshold.")
    else:
        rules = association_rules(freq, metric="confidence", min_threshold=CONFIG["apriori_min_confidence"])
        rules = rules[rules["lift"] >= CONFIG["apriori_min_lift"]]
        if rules.empty:
            st.write("No rules met the confidence and lift thresholds.")
        else:
            rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
            rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
            show = rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].sort_values("lift", ascending=False)
            st.dataframe(show.round(3), use_container_width=True)
            fig10, ax10 = plt.subplots(figsize=(8, 5))
            ax10.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", alpha=0.7)
            ax10.set_xlabel("Support")
            ax10.set_ylabel("Confidence")
            st.pyplot(fig10)

with tabs[5]:
    st.subheader("Derived priorities")
    bc = df[CONFIG["barangay_col"]].value_counts()
    st.write("Barangays with most flood records:")
    st.dataframe(bc.head(10).rename_axis("Barangay").reset_index(name="Records"), use_container_width=True)
    high_by_b = df[df["severity"] == "High"][CONFIG["barangay_col"]].value_counts()
    st.write("Barangays with most high-severity events:")
    st.dataframe(high_by_b.head(10).rename_axis("Barangay").reset_index(name="High-severity records"), use_container_width=True)
    high_by_m = df[(df["severity"] == "High") & df["month_num"].notna()]["month_num"].astype(int).value_counts().sort_index()
    high_by_m.index = [pd.Timestamp(2024, m, 1).strftime("%B") for m in high_by_m.index]
    st.write("Months with most high-severity events:")
    st.dataframe(high_by_m.rename_axis("Month").reset_index(name="High-severity records"), use_container_width=True)
    cause_share = df[CONFIG["cause_col"]].value_counts(normalize=True).mul(100).round(1)
    st.write("Share of records by weather system (%):")
    st.dataframe(cause_share.rename_axis("Weather System").reset_index(name="Share (%)"), use_container_width=True)
    st.write(f"Records span {int(df['year_num'].min())} to {int(df['year_num'].max())} where years are recorded.")

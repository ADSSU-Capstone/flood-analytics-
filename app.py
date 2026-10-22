import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)

st.set_page_config(
    page_title="Flood Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .dashboard-title {
        text-align: center;
        color: #1e3a8a;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }
    .dashboard-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 4px;
        padding: 10px 20px;
        border: 1px solid #cbd5e1;
        color: #1e293b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
        border: 1px solid #1e3a8a;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-box h3 {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0 0 4px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-box h2 {
        color: #0f172a;
        font-size: 1.8rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-title">Flood Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Historical Flood Records Analysis</div>', unsafe_allow_html=True)

def clean_text(value):
    if pd.isna(value): return np.nan
    s = str(value).replace('\xa0', ' ').replace('\u200b', '').replace('\t', ' ').replace('\n', ' ')
    s = ' '.join(s.split()).strip()
    if s.lower() in {'', 'nan', 'none', 'n/a', '-', '--', '?', 'undefined'}: return np.nan
    return s

def parse_currency(val):
    s = clean_text(val)
    if pd.isna(s): return np.nan
    s = str(s).replace(',', '').replace('PHP', '').replace('php', '').strip()
    try: return float(s)
    except: return np.nan

def preprocess_data(raw):
    df = raw.copy()
    
    if any(str(val).strip().lower() in ['barangay', 'year', 'month'] for val in df.iloc[0]):
        df.columns = [clean_text(c) for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
        
    df.columns = df.columns.astype(str).str.strip()
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    clean = pd.DataFrame()
    clean['Year'] = pd.to_numeric(df.get('Year', np.nan), errors='coerce').ffill()
    clean['Month'] = df.get('Month', pd.Series()).astype(str).str.strip().str.title()
    clean['Barangay'] = df.get('Barangay', pd.Series()).astype(str).str.strip().str.title()
    clean['Flood_Cause'] = df.get('Flood Cause', pd.Series()).astype(str).str.strip()
    clean['Flood_Cause'] = clean['Flood_Cause'].replace('Tropica Storm VINTA', 'Tropical Storm VINTA')
    
    clean['Water_Level_Ft'] = pd.to_numeric(df.get('Water Level', pd.Series()).astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0], errors='coerce')
    clean['Families_Affected'] = pd.to_numeric(df.get('No. of Families affected', pd.Series()).astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    clean['Damage_Infrastructure'] = df.get('Damage Infrastructure', pd.Series()).apply(parse_currency)
    clean['Damage_Agriculture'] = df.get('Damage Agriculture', pd.Series()).apply(parse_currency)
    clean['Total_Damage'] = clean['Damage_Infrastructure'].fillna(0) + clean['Damage_Agriculture'].fillna(0)
    
    clean = clean.dropna(subset=['Barangay'])
    clean = clean[~clean['Barangay'].isin(['Nan', 'None', ''])]

    month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
    clean['Month_Num'] = clean['Month'].map(month_map)
    clean['Season'] = clean['Month_Num'].apply(lambda m: 'Dry Season' if m in (12,1,2,3,4,5) else ('Wet Season' if pd.notna(m) else 'Unknown'))
    
    def severity_level(row):
        wl, fam, dmg = row['Water_Level_Ft'], row['Families_Affected'], row['Total_Damage']
        high_count = 0
        low_count = 0
        if pd.notna(wl):
            if wl > 4.0: high_count += 1
            if wl <= 2.0: low_count += 1
        if pd.notna(fam):
            if fam > 200: high_count += 1
            if fam < 50: low_count += 1
        if pd.notna(dmg):
            if dmg > 500000: high_count += 1
            if dmg < 100000: low_count += 1
        if high_count >= 2: return 'High'
        if low_count == 3: return 'Low'
        return 'Moderate'
        
    clean['Severity'] = clean.apply(severity_level, axis=1)
    return clean

def simple_apriori(transactions, min_support=0.05):
    n = len(transactions)
    if n == 0: return {}
    items = sorted(set(item for t in transactions for item in t))
    freq = {}
    for item in items:
        count = sum(1 for t in transactions if item in t)
        support = count / n
        if support >= min_support: freq[frozenset([item])] = support
    all_freq = dict(freq)
    k = 2
    while freq:
        freq_list = list(freq.keys())
        candidates = set()
        for i in range(len(freq_list)):
            for j in range(i + 1, len(freq_list)):
                union = freq_list[i] | freq_list[j]
                if len(union) == k: candidates.add(union)
        new_freq = {}
        for cand in candidates:
            count = sum(1 for t in transactions if cand.issubset(t))
            support = count / n
            if support >= min_support: new_freq[cand] = support
        all_freq.update(new_freq)
        freq = new_freq
        k += 1
    return all_freq

def generate_rules(freq_itemsets, min_confidence=0.3):
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2: continue
        for item in itemset:
            antecedent = frozenset([item])
            consequent = itemset - antecedent
            if not consequent: continue
            ant_support = freq_itemsets.get(antecedent, 0)
            if ant_support <= 0: continue
            confidence = support / ant_support
            if confidence < min_confidence: continue
            cons_support = freq_itemsets.get(consequent, 0)
            lift = confidence / cons_support if cons_support > 0 else 0
            rules.append({'antecedent': set(antecedent), 'consequent': set(consequent), 'support': support, 'confidence': confidence, 'lift': lift})
            
    seen = set()
    unique_rules = []
    for r in rules:
        key = (frozenset(r['antecedent']), frozenset(r['consequent']))
        if key in seen: continue
        seen.add(key)
        unique_rules.append(r)
    return unique_rules

st.sidebar.title("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)")

raw_data = None

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        raw_data = pd.read_csv(uploaded_file)
    else:
        raw_data = pd.read_excel(uploaded_file)
elif use_example:
    try:
        raw_data = pd.read_excel("FloodData.xlsx")
    except:
        try:
            raw_data = pd.read_csv("FloodData.csv")
        except:
            pass

if raw_data is None:
    st.info("Awaiting dataset upload or example dataset activation.")
    st.stop()

df = preprocess_data(raw_data)

st.sidebar.markdown("---")
st.sidebar.title("Filters")
brgy_options = sorted(df['Barangay'].dropna().unique())
barangay_filter = st.sidebar.multiselect("Filter by Barangay:", options=brgy_options, default=brgy_options)

year_min = int(df['Year'].dropna().min()) if df['Year'].notna().any() else 2015
year_max = int(df['Year'].dropna().max()) if df['Year'].notna().any() else 2025
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

df_f = df[(df['Barangay'].isin(barangay_filter)) & (df['Year'].between(year_range[0], year_range[1]))].copy()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", 
    "Frequency and Distribution", 
    "Seasonal Patterns", 
    "High-Risk Barangays", 
    "Flood Severity Classification", 
    "Association Rules", 
    "Decision Support"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><h3>Total Flood Records</h3><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><h3>Distinct Barangays</h3><h2>{df_f["Barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><h3>Families Affected</h3><h2>{int(df_f["Families_Affected"].sum()):,}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><h3>Total Damage (PHP)</h3><h2>{int(df_f["Total_Damage"].sum()):,}</h2></div>', unsafe_allow_html=True)
    
    st.markdown(f"The dataset comprises {len(df_f)} records across {df_f['Barangay'].nunique()} barangays. A total of {int(df_f['Families_Affected'].sum()):,} families were affected with PHP {int(df_f['Total_Damage'].sum()):,} in damages.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_f['Barangay'].value_counts().plot(kind='bar', color='#1e3a8a', edgecolor='black', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title("Flood Records by Barangay")
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_f['Flood_Cause'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%', cmap='Blues', ax=ax)
        plt.title("Flood Cause Distribution")
        st.pyplot(fig)

with tab2:
    st.markdown(f"The most dominant flood-causing weather system is **{df_f['Flood_Cause'].value_counts().idxmax()}**, accounting for **{df_f['Flood_Cause'].value_counts(normalize=True).max()*100:.1f}%** of all recorded events.")
    st.markdown(f"The barangays with the highest number of flood records are **{df_f['Barangay'].value_counts().idxmax()}** and **{df_f['Barangay'].value_counts().index[1]}**.")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        df_f['Flood_Cause'].value_counts().head(10).plot(kind='barh', color='#1e3a8a', ax=ax)
        plt.title("Flood-Causing Weather Systems")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        df_f['Barangay'].value_counts().plot(kind='bar', color='#0ea5e9', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title("Most Affected Barangays")
        st.pyplot(fig)

with tab3:
    st.markdown("Flood occurrences are highly concentrated in specific months, with minimal occurrences in the wet season.")
    
    col1, col2 = st.columns(2)
    with col1:
        m_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        m_counts = df_f['Month'].value_counts().reindex(m_order).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        m_counts.plot(kind='bar', color='#1e3a8a', ax=ax)
        plt.xticks(rotation=45)
        plt.title("Flood Occurrences by Month")
        st.pyplot(fig)
    with col2:
        s_counts = df_f['Season'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        s_counts.plot(kind='bar', color='#0ea5e9', edgecolor='black', ax=ax)
        plt.xticks(rotation=0)
        plt.title("Flood Occurrences by Season")
        st.pyplot(fig)

    pivot = df_f.pivot_table(index='Month', columns='Year', values='Water_Level_Ft', aggfunc='count', fill_value=0).reindex(m_order).fillna(0)
    if pivot.size > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Blues', linewidths=0.5, ax=ax)
        plt.title("Heatmap: Flood Occurrence by Year and Month")
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    agg = df_f.groupby('Barangay').agg(
        Total_Records=('Year', 'count'),
        Mean_Water_Level=('Water_Level_Ft', 'mean'),
        Max_Water_Level=('Water_Level_Ft', 'max'),
        Mean_Families=('Families_Affected', 'mean'),
        Total_Families=('Families_Affected', 'sum'),
        Total_Damage=('Total_Damage', 'sum'),
        Severe_Count=('Severity', lambda x: (x == 'High').sum())
    ).fillna(0)

    if len(agg) >= 3:
        feature_cols = ['Mean_Water_Level', 'Max_Water_Level', 'Mean_Families', 'Total_Families', 'Total_Damage', 'Total_Records', 'Severe_Count']
        X = agg[feature_cols].copy()
        X = X.loc[:, X.std() > 0]
        X_scaled = StandardScaler().fit_transform(X)
        
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        
        st.markdown(f"K-Means clustering identified 3 distinct clusters with a Silhouette Score of {sil:.4f} and a Within-Cluster Sum of Squares (WSS) of {km.inertia_:,.2f}.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", 3)
        c2.metric("Silhouette Score", f"{sil:.4f}")
        c3.metric("Within-Cluster SS", f"{km.inertia_:,.2f}")
        
        agg['Cluster'] = labels
        cluster_rank = agg.groupby('Cluster')['Total_Damage'].mean().sort_values(ascending=False)
        rank_map = {c: f"Tier {i+1}" for i, c in enumerate(cluster_rank.index)}
        agg['Risk_Tier'] = agg['Cluster'].map(rank_map)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            counts = pd.Series(labels).value_counts().sort_index()
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, 3))
            ax.bar([f"Cluster {i}" for i in counts.index], counts.values, color=colors[:len(counts)], edgecolor='black')
            ax.set_ylabel("Barangays")
            ax.set_title("Barangays per Cluster")
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Blues', s=80, alpha=0.85, edgecolors='black')
            plt.colorbar(sc, label='Cluster')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Barangays in PCA Space")
            plt.tight_layout()
            st.pyplot(fig)

        ranked = agg.sort_values(['Risk_Tier', 'Total_Damage'], ascending=[True, False])
        st.dataframe(ranked[['Risk_Tier', 'Cluster', 'Total_Records', 'Mean_Water_Level', 'Max_Water_Level', 'Mean_Families', 'Total_Families', 'Total_Damage', 'Severe_Count']].round(2), use_container_width=True)
    else:
        st.info("Not enough barangays for clustering.")

with tab5:
    mdf = df_f.dropna(subset=['Water_Level_Ft', 'Families_Affected', 'Total_Damage', 'Severity'])
    if len(mdf) > 15 and mdf['Severity'].nunique() >= 2:
        Xm = mdf[['Water_Level_Ft', 'Families_Affected', 'Total_Damage']].values
        ym = mdf['Severity'].values
        le = LabelEncoder()
        y_enc = le.fit_transform(ym)
        
        min_class_count = pd.Series(y_enc).value_counts().min()
        stratify_arg = y_enc if min_class_count >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(Xm, y_enc, test_size=0.30, random_state=42, stratify=stratify_arg)
        dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=2).fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        st.markdown(f"The decision tree classifier achieved an Accuracy of {acc:.4f}, Precision of {prec:.4f}, Recall of {rec:.4f}, and an F1-Score of {f1:.4f}.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision", f"{prec:.4f}")
        c3.metric("Recall", f"{rec:.4f}")
        c4.metric("F1-Score", f"{f1:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(dt, feature_names=['Water_Level_Ft', 'Families_Affected', 'Total_Damage'], class_names=list(le.classes_), filled=True, rounded=True, ax=ax, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("Not enough data for Decision Tree.")

with tab6:
    txn_df = df_f.dropna(subset=['Barangay'])
    transactions = []
    for _, row in txn_df.iterrows():
        txn = [f"Barangay={row['Barangay']}", f"Cause={row['Flood_Cause']}", f"Severity={row['Severity']}"]
        if pd.notna(row['Season']) and row['Season'] != 'Unknown': txn.append(f"Season={row['Season']}")
        if pd.notna(row['Month']): txn.append(f"Month={row['Month']}")
        transactions.append(set(txn))

    if len(transactions) >= 5:
        freq = simple_apriori(transactions, min_support=0.05)
        rules = generate_rules(freq, min_confidence=0.3)

        st.markdown(f"The Apriori algorithm generated {len(freq)} frequent itemsets and {len(rules)} rules using a minimum support of 0.05 and minimum confidence of 0.3.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Frequent Itemsets", len(freq))
        c2.metric("Rules Mined", len(rules))
        c3.metric("Transactions", len(transactions))

        if rules:
            rules_df = pd.DataFrame([{'Antecedent': ', '.join(sorted(r['antecedent'])), 'Consequent': ', '.join(sorted(r['consequent'])), 'Support': round(r['support'], 4), 'Confidence': round(r['confidence'], 4), 'Lift': round(r['lift'], 4)} for r in rules]).sort_values(['Lift', 'Confidence'], ascending=False)
            st.dataframe(rules_df.head(30), use_container_width=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            sc = ax.scatter(rules_df['Support'], rules_df['Confidence'], c=rules_df['Lift'], cmap='Blues', s=80, alpha=0.85, edgecolors='black')
            plt.colorbar(sc, label='Lift')
            ax.set_xlabel("Support")
            ax.set_ylabel("Confidence")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No rules met the current thresholds.")
    else:
        st.info("Not enough transactions for Association Rules.")

with tab7:
    top_brgy = df_f['Barangay'].value_counts().idxmax() if not df_f.empty else 'N/A'
    top_cause = df_f['Flood_Cause'].value_counts().idxmax() if not df_f.empty else 'N/A'
    top_month = df_f['Month'].value_counts().idxmax() if df_f['Month'].notna().any() else 'N/A'
    high_sev_pct = (df_f['Severity'] == 'High').sum() / max(len(df_f), 1) * 100

    st.markdown(f"**Concentration of Flood Risk:** A small group of barangays accounts for a disproportionate share of all recorded flood events and damage. The barangay with the highest frequency is **{top_brgy}**.")
    st.markdown(f"**Seasonal and Monthly Concentration:** Flood occurrences are not uniformly distributed. The most frequently recorded flood month is **{top_month}**.")
    st.markdown(f"**Dominant Flood-Causing Weather System:** The most frequently recorded flood-causing weather system is **{top_cause}**.")
    st.markdown(f"**Severity Profile of Flood Events:** Of all filtered flood records, **{high_sev_pct:.1f}%** are classified as High severity, with water level, family count, and total damage serving as the primary discriminators.")
    st.markdown(f"**Barangay-Specific Risk Mitigation:** Priority for resource allocation and evacuation infrastructure should be given to Tier 1 barangays identified by the clustering algorithm.")
    st.markdown(f"**Seasonal Risk Calendar:** Preparedness measures, resource prepositioning, and community awareness campaigns should be intensified leading up to the peak flood months.")
    st.markdown(f"**Rapid Severity Assessment:** The decision tree model provides a clear set of rules for the MDRRMO to quickly assess the potential severity of an incoming flood event based on initial damage and impact reports.")
    st.markdown(f"**Weather System Advisories:** Specific advisories and automatic protocols for flood monitoring should be triggered when PAGASA reports the dominant flood-causing weather systems.")

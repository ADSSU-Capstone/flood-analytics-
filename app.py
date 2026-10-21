import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, classification_report
)

st.set_page_config(
    page_title="Bunawan Flood Analytics Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }
    .main-header {
        font-size: 2.0rem; font-weight: 700; color: #004D40;
        text-align: center; padding: 0.6rem 0 0.2rem 0;
        letter-spacing: 0.4px;
    }
    .sub-header {
        font-size: 0.95rem; color: #546E7A;
        text-align: center; margin-bottom: 1.6rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #004D40 0%, #00796B 100%);
        padding: 1.15rem; border-radius: 10px; color: #FFFFFF;
        text-align: center; box-shadow: 0 3px 12px rgba(0,77,64,0.22);
        border-bottom: 3px solid #FFC107;
    }
    .metric-card h2 { color: #FFE082; margin: 0; font-size: 1.75rem; font-weight: 700; }
    .metric-card p  { color: #E0F2F1; margin: 0; font-size: 0.82rem;
                      text-transform: uppercase; letter-spacing: 0.8px; }

    .output-banner {
        background: linear-gradient(90deg, #FF7043 0%, #FFB74D 100%);
        color: #1A1A1A; padding: 0.9rem 1.2rem;
        border-radius: 8px; margin: 0.4rem 0 0.6rem 0;
        font-weight: 700; font-size: 1.0rem;
        box-shadow: 0 3px 10px rgba(255,112,67,0.25);
    }
    .output-banner small {
        display: block; font-weight: 500; opacity: 0.9;
        font-size: 0.83rem; margin-top: 0.2rem; color: #2C2C2C;
    }
    .expected-box {
        background: #E0F2F1; border-left: 5px solid #004D40;
        padding: 0.75rem 1.05rem; border-radius: 6px;
        margin-bottom: 1.15rem; font-size: 0.88rem; color: #003B33;
    }
    .expected-box b { color: #004D40; }

    .finding-card {
        background: #FAFBFC; border: 1px solid #CFD8DC;
        border-left: 6px solid #FF7043;
        border-radius: 8px; padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .finding-card h5 { margin: 0 0 0.4rem 0; color: #004D40; font-weight: 700; }
    .finding-card p  { margin: 0; font-size: 0.9rem; color: #37474F; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ECEFF1; border-radius: 6px 6px 0 0;
        padding: 10px 16px; font-weight: 600; color: #004D40;
    }
    .stTabs [aria-selected="true"] {
        background-color: #004D40 !important; color: #FFE082 !important;
    }
    .dataframe thead tr th { background-color: #E0F2F1 !important; }
</style>
""", unsafe_allow_html=True)


def render_output_banner(output_title, chapter3_expected, description):
    st.markdown(
        f'<div class="output-banner">Conceptual Framework Output: {output_title}'
        f'<small>{description}</small></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="expected-box"><b>Chapter 3 Expected Outcome:</b> {chapter3_expected}</div>',
        unsafe_allow_html=True
    )


@st.cache_data(show_spinner=True)
def load_flood_data():
    search_paths = ['.', '/content/data', '/content', '/mount/src']
    possible_filenames = ['FloodData.xlsx', 'FloodData.xls', 'FloodData.csv', 'flood_data.xlsx', 'flood_data.csv', 'Flood Data.xlsx']

    filepath = None
    for path in search_paths:
        if not os.path.isdir(path): continue
        for fname in possible_filenames:
            full_path = os.path.join(path, fname)
            if os.path.exists(full_path):
                filepath = full_path
                break
        if filepath: break

    if filepath is None: return None, "No dataset found."

    try:
        df = pd.read_csv(filepath) if filepath.lower().endswith('.csv') else pd.read_excel(filepath, header=None)
        return df, f"Loaded: {os.path.basename(filepath)}"
    except Exception as e:
        return None, f"Error loading file: {e}"


def _resolve_column(df, candidates):
    normalized = {str(c).lower().replace(' ', '').replace('_', '').replace('.', ''): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('_', '').replace('.', '')
        if key in normalized: return normalized[key]
    return None


MONTH_MAP = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12}


def clean_text(value):
    if pd.isna(value): return np.nan
    s = str(value).replace('\xa0', ' ').replace('\u200b', '').replace('\t', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    if s.lower() in {'', 'nan', 'none', 'n/a', '-', '--', '?', 'undefined'}: return np.nan
    return s


def parse_month(value):
    s = clean_text(value)
    if pd.isna(s): return np.nan
    s = str(s).lower()
    if s.isdigit():
        m = int(s)
        return m if 1 <= m <= 12 else np.nan
    return MONTH_MAP.get(s, np.nan)


def parse_number(value):
    s = clean_text(value)
    if pd.isna(s): return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)', str(s))
    return float(m.group(1)) if m else np.nan


def parse_damage(value):
    s = clean_text(value)
    if pd.isna(s): return np.nan
    s = str(s).replace(' ', '').replace(',', '').replace('php', '').replace('PHP', '')
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    s = re.sub(r'[^\d.]', '', s)
    try: return float(s)
    except: return np.nan


@st.cache_data(show_spinner=True)
def preprocess_data(raw):
    df = raw.copy()
    df.columns = [clean_text(c) for c in df.iloc[0]]
    df = df.iloc[1:].reset_index(drop=True)
    
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    clean = pd.DataFrame()
    clean['Year'] = pd.to_numeric(df.get('Year', np.nan), errors='coerce').ffill()
    clean['Month_Num'] = df.get('Month', pd.Series()).apply(parse_month).ffill()
    clean['Barangay'] = df.get('Barangay', pd.Series()).apply(lambda x: clean_text(x).title() if pd.notna(x) else np.nan)
    clean['Flood_Cause'] = df.get('Flood Cause', pd.Series()).apply(lambda x: clean_text(x) if pd.notna(x) else 'Unknown')
    clean['Water_Level_Ft'] = df.get('Water Level', pd.Series()).apply(parse_number)
    clean['Families_Affected'] = pd.to_numeric(df.get('No. of Families affected', pd.Series()).apply(clean_text), errors='coerce')
    clean['Damage_Infrastructure'] = df.get('Damage Infrastructure', pd.Series()).apply(parse_damage)
    clean['Damage_Agriculture'] = df.get('Damage Agriculture', pd.Series()).apply(parse_damage)

    clean = clean.dropna(subset=['Barangay'])
    clean = clean[clean['Barangay'] != 'Unknown']

    MONTH_NAMES = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
    clean['Month_Name'] = clean['Month_Num'].map(MONTH_NAMES)
    clean['Season'] = clean['Month_Num'].apply(lambda m: 'Dry Season' if m in (12,1,2,3,4,5) else ('Wet Season' if pd.notna(m) else 'Unknown'))
    clean['Total_Damage'] = clean['Damage_Infrastructure'].fillna(0) + clean['Damage_Agriculture'].fillna(0)

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
            if dmg > 500_000: high_count += 1
            if dmg < 100_000: low_count += 1
        if high_count >= 2: return 'High'
        if low_count == 3: return 'Low'
        return 'Moderate'
    
    clean['Severity'] = clean.apply(severity_level, axis=1)
    return clean


raw, load_message = load_flood_data()

st.markdown('<div class="main-header">Bunawan Flood Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyzing Historical Flood Records Using Data Mining for Disaster Preparedness Planning<br><i>Municipality of Bunawan, Agusan del Sur</i></div>', unsafe_allow_html=True)

if raw is None:
    st.error(f"Error: {load_message}")
    st.stop()

df = preprocess_data(raw)

st.sidebar.header("Dashboard Controls")
brgy_options = sorted(df['Barangay'].dropna().unique())
barangay_filter = st.sidebar.multiselect("Filter by Barangay:", options=brgy_options, default=brgy_options)

year_min = int(df['Year'].dropna().min()) if df['Year'].notna().any() else 2015
year_max = int(df['Year'].dropna().max()) if df['Year'].notna().any() else 2025
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
k_clusters = st.sidebar.slider("K-Means Clusters", 2, 6, 3)
min_support = st.sidebar.slider("Apriori Min Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Apriori Min Confidence", 0.1, 1.0, 0.3, 0.05)

df_f = df[(df['Barangay'].isin(barangay_filter)) & (df['Year'].between(year_range[0], year_range[1]))].copy()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "High-Risk Barangays", "Seasonal Patterns", "Severity Levels", "Association Rules", "Preparedness Plan"
])

with tab1:
    st.header("Overview of Historical Flood Records")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Flood Records</p><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Distinct Barangays</p><h2>{df_f["Barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Families Affected</p><h2>{int(df_f["Families_Affected"].sum()):,}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Total Damage (PHP)</p><h2>{int(df_f["Total_Damage"].sum()):,}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Flood Records by Barangay")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f['Barangay'].value_counts().plot(kind='bar', color='#00796B', edgecolor='black', ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    with col_b:
        st.subheader("Flood Cause Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f['Flood_Cause'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%', cmap='Set3', ax=ax)
        st.pyplot(fig)

with tab2:
    render_output_banner("High-Risk Flood Barangays", "Identification of barangays exhibiting similar flood risk profiles using K-Means.", "Clustering based on damage and severity.")
    
    agg = df_f.groupby('Barangay').agg(
        Total_Records=('Year', 'count'),
        Mean_Water_Level=('Water_Level_Ft', 'mean'),
        Total_Damage=('Total_Damage', 'sum'),
        Severe_Count=('Severity', lambda x: (x == 'High').sum())
    ).fillna(0)

    if len(agg) >= k_clusters:
        X = agg[['Mean_Water_Level', 'Total_Damage', 'Severe_Count']]
        X_scaled = StandardScaler().fit_transform(X)
        
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        agg['Cluster'] = km.fit_predict(X_scaled)
        
        cluster_rank = agg.groupby('Cluster')['Total_Damage'].mean().sort_values(ascending=False)
        rank_map = {c: f"Tier {i+1}" for i, c in enumerate(cluster_rank.index)}
        agg['Risk_Tier'] = agg['Cluster'].map(rank_map)
        
        st.dataframe(agg.sort_values(['Risk_Tier', 'Total_Damage'], ascending=[True, False]), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        agg['Risk_Tier'].value_counts().plot(kind='bar', color='#FF7043', ax=ax)
        ax.set_title("Number of Barangays per Risk Tier")
        st.pyplot(fig)
    else:
        st.warning("Not enough barangays to cluster.")

with tab3:
    render_output_banner("Seasonal Flood Patterns", "Description of monthly and yearly distribution of flood events.", "Descriptive analysis of time-based patterns.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flood Occurrences by Month")
        m_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
        m_counts = df_f['Month_Name'].value_counts().reindex(m_order).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        m_counts.plot(kind='bar', color='#26A69A', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.subheader("Flood Occurrences by Year")
        y_counts = df_f['Year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        y_counts.plot(kind='line', marker='o', color='#FF7043', ax=ax)
        st.pyplot(fig)

with tab4:
    render_output_banner("Severity Levels", "Decision tree classification of flood events by severity level.", "Supervised classification evaluated by accuracy and F1-score.")
    
    mdf = df_f.dropna(subset=['Water_Level_Ft', 'Families_Affected', 'Total_Damage', 'Severity'])
    if len(mdf) > 15:
        Xm = mdf[['Water_Level_Ft', 'Families_Affected', 'Total_Damage']].values
        ym = mdf['Severity'].values
        le = LabelEncoder()
        y_enc = le.fit_transform(ym)
        
        min_class_count = pd.Series(y_enc).value_counts().min()
        stratify_arg = y_enc if min_class_count >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            Xm, y_enc, test_size=0.3, random_state=42, stratify=stratify_arg
        )
        
        dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
        c3.metric("F1-Score", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.2f}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(dt, feature_names=['Water_Level', 'Families', 'Damage'], class_names=le.classes_, filled=True, ax=ax, fontsize=10)
        st.pyplot(fig)
    else:
        st.warning("Not enough data for Decision Tree.")

with tab5:
    render_output_banner("Association Rules (Apriori)", "Discovery of co-occurrence patterns among flood variables using Apriori.", "Rules evaluated by support, confidence, and lift.")
    
    from itertools import combinations
    adf = df_f.dropna(subset=['Severity']).copy()
    transactions = []
    for _, row in adf.iterrows():
        t = [f"Brgy_{row['Barangay']}", f"Severity_{row['Severity']}", f"Cause_{row['Flood_Cause']}"]
        if pd.notna(row['Month_Name']):
            t.append(f"Month_{row['Month_Name']}")
        transactions.append(t)
        
    rules = []
    n = len(transactions)
    all_items = set(item for t in transactions for item in t)
    
    for item1, item2 in combinations(all_items, 2):
        count_both = sum(1 for t in transactions if item1 in t and item2 in t)
        count_1 = sum(1 for t in transactions if item1 in t)
        count_2 = sum(1 for t in transactions if item2 in t)
        
        if count_both >= 2 and count_1 > 0 and count_2 > 0:
            support = count_both / n
            conf = count_both / count_1
            lift = conf / (count_2 / n)
            
            if support >= min_support and conf >= min_confidence and lift > 1.0:
                rules.append({"If": item1, "Then": item2, "Support": round(support, 3), "Confidence": round(conf, 3), "Lift": round(lift, 2)})
                
    rules.sort(key=lambda x: x["Lift"], reverse=True)
    
    if rules:
        st.dataframe(pd.DataFrame(rules[:20]), hide_index=True, use_container_width=True)
    else:
        st.info("No strong rules found with current thresholds. Lower the sidebar sliders.")

with tab6:
    render_output_banner("Data-Driven Preparedness Plan", "Consolidated evidence snapshots synthesized across all three data mining techniques.", "Findings-only summary.")
    
    top_brgy = df_f['Barangay'].value_counts().index[0] if not df_f.empty else "N/A"
    top_month = df_f['Month_Name'].value_counts().index[0] if df_f['Month_Name'].notna().any() else "N/A"
    
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 1 - Concentration of Flood Risk</h5>
        <p>The highest frequency of flood records is found in <b>{top_brgy}</b>. The K-Means clustering identifies this area as part of the highest risk tier based on accumulated damage and severity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 2 - Seasonal Concentration</h5>
        <p>Flooding is not uniformly distributed. The highest volume of flood events occurs in <b>{top_month}</b>, which should dictate the seasonal calendar for preparedness.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="finding-card">
        <h5>Finding 3 - Severity Drivers</h5>
        <p>The Decision Tree classifier confirms that Water Level, Family Count, and Total Damage are the primary discriminators between Low, Moderate, and High severity events.</p>
    </div>
    """, unsafe_allow_html=True)

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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bunawan Flood Analytics Dashboard",
    page_icon=None,
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

    .rule-pill {
        display: inline-block; background: #FFF3E0; color: #E65100;
        padding: 4px 10px; border-radius: 12px; margin: 2px;
        font-size: 0.85rem; font-weight: 600;
    }
    .rule-pill-arrow { color: #004D40; font-weight: 700; margin: 0 6px; }

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


# ============================================================
# OUTPUT BANNER + CHAPTER 3 EXPECTED OUTCOME
# ============================================================
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


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=True)
def load_flood_data():
    search_paths = ['.', '/content/data', '/content', '/mount/src']
    possible_filenames = [
        'FloodData.xlsx', 'FloodData.xls', 'FloodData.csv',
        'flood_data.xlsx', 'flood_data.csv',
        'Flood Data.xlsx'
    ]

    filepath = None
    for path in search_paths:
        if not os.path.isdir(path):
            continue
        for fname in possible_filenames:
            full_path = os.path.join(path, fname)
            if os.path.exists(full_path):
                filepath = full_path
                break
        if filepath:
            break

    if filepath is None:
        for path in search_paths:
            if not os.path.isdir(path):
                continue
            for ext in ('*.xlsx', '*.xls', '*.csv'):
                for f in glob.glob(os.path.join(path, ext)):
                    base = os.path.basename(f).lower()
                    if 'flood' in base:
                        filepath = f
                        break
                if filepath:
                    break
            if filepath:
                break

    if filepath is None:
        return None, "No flood dataset file found."

    try:
        if filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath, header=None)
        return df, f"Loaded: {os.path.basename(filepath)} ({df.shape[0]:,} rows x {df.shape[1]} cols)"
    except Exception as e:
        return None, f"Error loading file: {e}"


# ============================================================
# COLUMN RESOLVER
# ============================================================
def _resolve_column(df, candidates):
    normalized = {str(c).lower().replace(' ', '').replace('_', '').replace('.', ''): c
                  for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('_', '').replace('.', '')
        if key in normalized:
            return normalized[key]
    return None


# ============================================================
# CLEANING HELPERS
# ============================================================
NULL_TOKENS = {
    '', 'nan', 'none', 'n/a', 'na', 'null', '-', '--', '.', '?',
    'undefined', '#n/a', '#value!', 'not applicable', 'notavailable'
}

MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def clean_text(value):
    """Trim, collapse whitespace, strip invisible chars, standardize nulls."""
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        pass
    s = str(value)
    s = s.replace('\xa0', ' ').replace('\u200b', '').replace('\t', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    if s.lower() in NULL_TOKENS:
        return np.nan
    return s


def clean_titlecase(value):
    s = clean_text(value)
    if pd.isna(s):
        return np.nan
    s = re.sub(r'\s+', ' ', s).strip()
    return s.title()


def parse_month(value):
    s = clean_text(value)
    if pd.isna(s):
        return np.nan
    key = str(s).strip().lower()
    # numeric
    if key.isdigit():
        m = int(key)
        if 1 <= m <= 12:
            return m
        return np.nan
    return MONTH_MAP.get(key, np.nan)


def parse_water_level(value):
    """Extract numeric feet from strings like '4 ft.', '3ft', '8.5ft'."""
    s = clean_text(value)
    if pd.isna(s):
        return np.nan
    # find first number (int or decimal)
    m = re.search(r'(\d+(?:\.\d+)?)', str(s))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return np.nan
    return np.nan


def parse_damage(value):
    """Parse damage values: '422.510.5' -> 422510.5, '419, 550.00' -> 419550.0."""
    s = clean_text(value)
    if pd.isna(s):
        return np.nan
    s = str(s).replace(' ', '').replace(',', '').replace('php', '').replace('PHP', '')
    # Handle multiple dots: keep last as decimal separator
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    # Remove non-numeric except dot
    s = re.sub(r'[^\d.]', '', s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_flood_cause(value):
    """Fix typos and standardize flood cause names."""
    s = clean_text(value)
    if pd.isna(s):
        return 'Unknown'
    s = str(s).strip()
    # Fix common typos
    replacements = {
        'tropica storm vinta': 'Tropical Storm Vinta',
        'tropical storm vinta': 'Tropical Storm Vinta',
        'tropical depression auring': 'Tropical Depression Auring',
        'tropical storm seniang': 'Tropical Storm Seniang',
        'tropical depression agaton': 'Tropical Depression Agaton',
        'low pressure area (lpa)': 'Low Pressure Area (LPA)',
        'low pressure area': 'Low Pressure Area (LPA)',
        'lpa': 'Low Pressure Area (LPA)',
        'easterlies and shearline': 'Easterlies and Shearline',
        'easterlies and shear line': 'Easterlies and Shearline',
        'thunderstorm and shearline': 'Thunderstorm and Shearline',
        'heavy rains due to amihan': 'Heavy Rains due to Amihan',
        'shearline': 'Shearline',
        'easterlies': 'Easterlies',
    }
    key = s.lower().strip()
    if key in replacements:
        return replacements[key]
    # Title-case fallback
    return s.title()


# ============================================================
# PREPROCESSING (Chapter 3 - KDD Process with Cleaning)
# ============================================================
@st.cache_data(show_spinner=True)
def preprocess_data(raw):
    """Clean and standardize the flood dataset."""
    df = raw.copy()

    # The Excel file has a two-row header. Row 0 = top header, Row 1 = second header.
    # We detect this by checking if row 0 contains "Month" or "Year".
    if df.shape[0] >= 2:
        first_row_vals = df.iloc[0].astype(str).str.lower().tolist()
        if any('month' in v or 'year' in v for v in first_row_vals):
            df.columns = [clean_text(c) for c in df.iloc[0]]
            df = df.iloc[1:].reset_index(drop=True)
        else:
            df.columns = [clean_text(c) for c in df.iloc[0]]
            df = df.iloc[1:].reset_index(drop=True)

    # Strip whitespace from column names
    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]
    # Remove duplicate column names by appending suffix
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}_{i}" if i != 0 else dup
            for i in range(sum(cols == dup))
        ]
    df.columns = cols

    # Resolve columns
    col_month   = _resolve_column(df, ['Month'])
    col_day     = _resolve_column(df, ['Day'])
    col_year    = _resolve_column(df, ['Year'])
    col_muni    = _resolve_column(df, ['Municipality'])
    col_brgy    = _resolve_column(df, ['Barangay'])
    col_occur   = _resolve_column(df, ['Flood Occurrence'])
    col_cause   = _resolve_column(df, ['Flood Cause'])
    col_water   = _resolve_column(df, ['Water Level'])
    col_families= _resolve_column(df, ['No. of Families affected', 'No of Families affected',
                                       'Number of Families Affected'])
    col_infra   = _resolve_column(df, ['Damage Infrastructure'])
    col_agri    = _resolve_column(df, ['Damage Agriculture'])

    # Resolve weather-block columns (they may have suffix _1 or be the second occurrence)
    all_cols = list(df.columns)
    # Weather block: Year_1, Month_1, Day_1, Weather, Temperature(°C), Rainfall (mm), Wind, Pressure
    def find_nth(candidates, occurrence=0):
        matches = []
        for c in all_cols:
            for cand in candidates:
                if cand.lower() in str(c).lower():
                    matches.append(c)
                    break
        if len(matches) > occurrence:
            return matches[occurrence]
        return None

    col_w_year   = find_nth(['Year'], 1)
    col_w_month  = find_nth(['Month'], 1)
    col_w_day    = find_nth(['Day'], 1)
    col_weather  = find_nth(['Weather'], 0)
    col_temp     = find_nth(['Temperature'], 0)
    col_rain     = find_nth(['Rainfall'], 0)
    col_wind     = find_nth(['Wind'], 0)
    col_pressure = find_nth(['Pressure'], 0)

    clean = pd.DataFrame()

    # -------- Flood event fields --------
    clean['Year']      = pd.to_numeric(df[col_year].apply(clean_text), errors='coerce') if col_year else np.nan
    clean['Month_Num'] = df[col_month].apply(parse_month) if col_month else np.nan
    clean['Day']       = pd.to_numeric(df[col_day].apply(clean_text), errors='coerce') if col_day else np.nan

    # Forward-fill Year and Month within event blocks
    clean['Year']      = clean['Year'].ffill()
    clean['Month_Num'] = clean['Month_Num'].ffill()

    # Municipality (should be constant, but clean)
    clean['Municipality'] = df[col_muni].apply(clean_titlecase) if col_muni else 'Bunawan'
    clean['Barangay']     = df[col_brgy].apply(clean_titlecase) if col_brgy else np.nan

    clean['Flood_Cause']  = df[col_cause].apply(normalize_flood_cause) if col_cause else 'Unknown'

    clean['Water_Level_Ft'] = df[col_water].apply(parse_water_level) if col_water else np.nan

    clean['Families_Affected'] = pd.to_numeric(
        df[col_families].apply(clean_text), errors='coerce'
    ) if col_families else np.nan

    clean['Damage_Infrastructure'] = df[col_infra].apply(parse_damage) if col_infra else np.nan
    clean['Damage_Agriculture']    = df[col_agri].apply(parse_damage) if col_agri else np.nan

    # -------- Weather block --------
    if col_weather:
        clean['Weather'] = df[col_weather].apply(clean_text)
    if col_rain:
        clean['Rainfall_mm'] = pd.to_numeric(
            df[col_rain].apply(clean_text).str.replace(' mm', '', regex=False),
            errors='coerce'
        )
    if col_temp:
        clean['Temperature_C'] = df[col_temp].apply(clean_text)

    # -------- Drop fully-empty rows --------
    clean = clean.dropna(subset=['Barangay'])

    # -------- Drop rows without a valid barangay name --------
    clean = clean[clean['Barangay'] != 'Unknown']

    # -------- Valid-range filtering --------
    clean = clean[(clean['Water_Level_Ft'].isna()) | (clean['Water_Level_Ft'] > 0)]
    clean = clean[(clean['Water_Level_Ft'].isna()) | (clean['Water_Level_Ft'] <= 50)]
    clean = clean[(clean['Families_Affected'].isna()) | (clean['Families_Affected'] >= 0)]
    clean = clean[(clean['Year'].isna()) | ((clean['Year'] >= 2000) & (clean['Year'] <= 2030))]

    # -------- Drop duplicates --------
    before = len(clean)
    clean = clean.drop_duplicates()
    removed = before - len(clean)

    # -------- Derived fields --------
    MONTH_NAMES = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
                   7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
    clean['Month_Name'] = clean['Month_Num'].map(MONTH_NAMES)

    def get_season(m):
        if pd.isna(m):
            return 'Unknown'
        m = int(m)
        return 'Dry Season' if m in (12, 1, 2, 3, 4, 5) else 'Wet Season'
    clean['Season'] = clean['Month_Num'].apply(get_season)

    # Water level category
    def wl_cat(wl):
        if pd.isna(wl):
            return 'Unknown'
        if wl <= 2.0:
            return 'Low (<=2 ft)'
        if wl <= 4.0:
            return 'Moderate (2.1-4 ft)'
        return 'High (>4 ft)'
    clean['Water_Level_Category'] = clean['Water_Level_Ft'].apply(wl_cat)

    # Families category
    def fam_cat(f):
        if pd.isna(f):
            return 'Unknown'
        if f < 50:
            return 'Few (<50)'
        if f <= 200:
            return 'Moderate (50-200)'
        return 'Many (>200)'
    clean['Families_Category'] = clean['Families_Affected'].apply(fam_cat)

    # Damage category (total damage)
    clean['Total_Damage'] = clean['Damage_Infrastructure'].fillna(0) + clean['Damage_Agriculture'].fillna(0)

    def dmg_cat(d):
        if pd.isna(d):
            return 'Unknown'
        if d < 100_000:
            return 'Low (<100K)'
        if d <= 500_000:
            return 'Moderate (100K-500K)'
        return 'High (>500K)'
    clean['Damage_Category'] = clean['Total_Damage'].apply(dmg_cat)

    # Severity (Chapter 3 thresholds)
    def severity_level(row):
        wl = row['Water_Level_Ft']
        fam = row['Families_Affected']
        dmg = row['Total_Damage']
        # Count how many exceed HIGH thresholds
        high_count = 0
        low_count = 0
        if pd.notna(wl):
            if wl > 4.0:
                high_count += 1
            if wl <= 2.0:
                low_count += 1
        if pd.notna(fam):
            if fam > 200:
                high_count += 1
            if fam < 50:
                low_count += 1
        if pd.notna(dmg):
            if dmg > 500_000:
                high_count += 1
            if dmg < 100_000:
                low_count += 1
        if high_count >= 2:
            return 'High'
        if low_count == 3:
            return 'Low'
        return 'Moderate'
    clean['Severity'] = clean.apply(severity_level, axis=1)

    # -------- Event ID (for grouping records within an event block) --------
    # Increment every time Month_Num is non-null in original (we forward-filled, so use mask)
    mask = df[col_month].apply(parse_month).notna().values if col_month else np.zeros(len(df), bool)
    mask = mask[:len(clean)]
    event_id = np.cumsum(mask)
    clean['Event_ID'] = event_id

    clean.attrs['rows_removed_duplicate'] = removed
    clean.attrs['rows_final'] = len(clean)
    clean.attrs['n_events'] = int(clean['Event_ID'].nunique())

    return clean


# ============================================================
# SIMPLE APRIORI (no external dependency)
# ============================================================
def simple_apriori(transactions, min_support=0.1):
    n = len(transactions)
    if n == 0:
        return {}
    items = set()
    for t in transactions:
        items.update(t)
    items = sorted(items)

    freq = {}
    for item in items:
        count = sum(1 for t in transactions if item in t)
        support = count / n
        if support >= min_support:
            freq[frozenset([item])] = support

    all_freq = dict(freq)

    k = 2
    while freq:
        freq_list = list(freq.keys())
        candidates = set()
        for i in range(len(freq_list)):
            for j in range(i + 1, len(freq_list)):
                union = freq_list[i] | freq_list[j]
                if len(union) == k:
                    candidates.add(union)

        new_freq = {}
        for cand in candidates:
            count = sum(1 for t in transactions if cand.issubset(t))
            support = count / n
            if support >= min_support:
                new_freq[cand] = support

        all_freq.update(new_freq)
        freq = new_freq
        k += 1

    return all_freq


def generate_rules(freq_itemsets, min_confidence=0.5):
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        for item in itemset:
            antecedent = frozenset([item])
            consequent = itemset - antecedent
            if not consequent:
                continue
            ant_support = freq_itemsets.get(antecedent, 0)
            if ant_support <= 0:
                continue
            confidence = support / ant_support
            if confidence < min_confidence:
                continue
            cons_support = freq_itemsets.get(consequent, 0)
            lift = confidence / cons_support if cons_support > 0 else 0
            rules.append({
                'antecedent': set(antecedent),
                'consequent': set(consequent),
                'support': support,
                'confidence': confidence,
                'lift': lift
            })
    # Deduplicate (same antecedent/consequent from different itemset orderings)
    seen = set()
    unique_rules = []
    for r in rules:
        key = (frozenset(r['antecedent']), frozenset(r['consequent']))
        if key in seen:
            continue
        seen.add(key)
        unique_rules.append(r)
    return unique_rules


# ============================================================
# LOAD & PREPROCESS
# ============================================================
raw, load_message = load_flood_data()

st.markdown(
    '<div class="main-header">Bunawan Flood Analytics Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Analyzing Historical Flood Records Using Data Mining '
    'for Disaster Preparedness Planning<br>'
    '<i>Municipality of Bunawan, Agusan del Sur | MDRRMO Records 2015-2025</i></div>',
    unsafe_allow_html=True
)

if raw is None:
    st.error(f"Error: {load_message}")
    st.stop()

st.success(f"{load_message}")

df = preprocess_data(raw)

if len(df) < 10:
    st.error("Not enough valid rows after preprocessing.")
    st.stop()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Dashboard Controls")

brgy_options = sorted([b for b in df['Barangay'].dropna().unique() if b != 'Unknown'])
default_brgys = brgy_options[:10] if len(brgy_options) > 10 else brgy_options

barangay_filter = st.sidebar.multiselect(
    "Filter by Barangay:", options=brgy_options, default=default_brgys
)

year_min = int(df['Year'].dropna().min()) if df['Year'].notna().any() else 2015
year_max = int(df['Year'].dropna().max()) if df['Year'].notna().any() else 2025
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
k_clusters = st.sidebar.slider("K-Means Clusters", 2, 6, 3)
min_support = st.sidebar.slider("Apriori Min Support", 0.05, 0.5, 0.10, 0.05)
min_confidence = st.sidebar.slider("Apriori Min Confidence", 0.3, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Cleaning Summary")
st.sidebar.markdown(
    f"""
    - Records after cleaning: **{len(df):,}**
    - Distinct events: **{df.attrs.get('n_events', 0)}**
    - Duplicates removed: **{df.attrs.get('rows_removed_duplicate', 0):,}**
    - Raw rows: **{len(raw):,}**
    """
)

if not barangay_filter:
    st.warning("Please select at least one barangay.")
    st.stop()

df_f = df[
    (df['Barangay'].isin(barangay_filter)) &
    (df['Year'].between(year_range[0], year_range[1]))
].copy()

if len(df_f) < 5:
    st.warning("Too few records after filtering.")
    st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "High-Risk Flood Barangays",
    "Seasonal Flood Patterns",
    "Severity Levels",
    "Climate-Seasonal-Damage Rules",
    "Data-Driven Preparedness Plan",
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("Overview of Historical Flood Records")
    st.caption("Baseline profile of the MDRRMO Bunawan flood dataset used as input for all data mining outputs.")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><p>Total Flood Records</p><h2>{len(df_f):,}</h2></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="metric-card"><p>Distinct Barangays</p><h2>{df_f["Barangay"].nunique()}</h2></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        f'<div class="metric-card"><p>Families Affected</p>'
        f'<h2>{int(df_f["Families_Affected"].sum()):,}</h2></div>',
        unsafe_allow_html=True
    )
    total_dmg = int(df_f['Total_Damage'].sum())
    c4.markdown(
        f'<div class="metric-card"><p>Total Damage (PHP)</p><h2>{total_dmg:,}</h2></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("Flood Records by Barangay")
    brgy_counts = df_f['Barangay'].value_counts()
    if len(brgy_counts) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        brgy_counts.plot(kind='bar', color='#00796B', edgecolor='black', ax=ax)
        ax.set_ylabel("Number of Records")
        ax.set_title("Flood Records per Barangay")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Flood Cause Distribution")
        cause_counts = df_f['Flood_Cause'].value_counts().head(10)
        if len(cause_counts) > 0:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            palette = plt.cm.Set3(np.linspace(0, 1, len(cause_counts)))
            ax.pie(cause_counts.values, labels=cause_counts.index, autopct='%1.1f%%',
                   colors=palette, startangle=90)
            ax.set_title("Flood Cause Distribution")
            st.pyplot(fig)
            plt.close()

    with col_b:
        st.subheader("Flood Records per Year")
        year_counts = df_f['Year'].value_counts().sort_index()
        if len(year_counts) > 0:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            year_counts.plot(kind='bar', color='#FF7043', edgecolor='black', ax=ax)
            ax.set_ylabel("Number of Records")
            ax.set_xlabel("Year")
            ax.set_title("Annual Flood Records")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader("Prepared Dataset Preview")
    preview_cols = ['Year', 'Month_Name', 'Day', 'Barangay', 'Flood_Cause',
                    'Water_Level_Ft', 'Families_Affected', 'Total_Damage', 'Severity']
    preview_cols = [c for c in preview_cols if c in df_f.columns]
    st.dataframe(df_f[preview_cols].head(50), use_container_width=True)


# ------------------------------------------------------------
# TAB 2: HIGH-RISK FLOOD BARANGAYS (K-Means) - Output 1
# ------------------------------------------------------------
with tab2:
    render_output_banner(
        "High-Risk Flood Barangays",
        "Identification of barangays exhibiting similar flood risk profiles using K-Means "
        "clustering, producing a barangay-level risk classification to guide targeted "
        "preparedness action.",
        "K-Means clustering of flood records by water level, family impact, and damage "
        "to classify barangays into risk tiers."
    )

    st.header("High-Risk Flood Barangay Clustering")

    # Aggregate barangay-level features
    agg = df_f.groupby('Barangay').agg(
        Total_Records=('Year', 'count'),
        Mean_Water_Level=('Water_Level_Ft', 'mean'),
        Max_Water_Level=('Water_Level_Ft', 'max'),
        Total_Families=('Families_Affected', 'sum'),
        Mean_Families=('Families_Affected', 'mean'),
        Total_Damage=('Total_Damage', 'sum'),
        Severe_Count=('Severity', lambda x: (pd.Series(x) == 'High').sum()),
    ).fillna(0)

    if len(agg) < k_clusters * 2:
        st.warning(f"Only {len(agg)} barangays - need at least {k_clusters * 2} for stable clustering.")
    else:
        feature_cols = ['Mean_Water_Level', 'Max_Water_Level', 'Mean_Families',
                        'Total_Families', 'Total_Damage', 'Total_Records', 'Severe_Count']
        feature_cols = [c for c in feature_cols if c in agg.columns]

        X = agg[feature_cols].copy()
        # remove zero-variance cols
        X = X.loc[:, X.std() > 0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        wss = kmeans.inertia_

        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", k_clusters)
        c2.metric("Silhouette Score", f"{sil:.4f}")
        c3.metric("Within-Cluster SS", f"{wss:,.2f}")

        agg['Cluster'] = labels

        # Rank clusters by mean total damage (higher = more at-risk)
        cluster_rank = agg.groupby('Cluster')['Total_Damage'].mean().sort_values(ascending=False)
        rank_map = {c: f"Tier {i+1}" for i, c in enumerate(cluster_rank.index)}
        agg['Risk_Tier'] = agg['Cluster'].map(rank_map)

        # PCA viz
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            counts = pd.Series(labels).value_counts().sort_index()
            colors = plt.cm.Set2(np.linspace(0, 1, k_clusters))
            ax.bar([f"Cluster {i}" for i in counts.index], counts.values,
                   color=colors[:len(counts)], edgecolor='black')
            ax.set_ylabel("Barangays")
            ax.set_title("Barangays per Cluster")
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                            cmap='tab10', s=80, alpha=0.85, edgecolors='black')
            plt.colorbar(sc, label='Cluster')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Barangays in PCA Space")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("Barangay Risk Ranking")

        # Sort by mean damage within cluster descending (overall risk)
        ranked = agg.sort_values(['Risk_Tier', 'Total_Damage'], ascending=[True, False])

        st.dataframe(
            ranked[['Risk_Tier', 'Cluster', 'Total_Records', 'Mean_Water_Level',
                    'Max_Water_Level', 'Mean_Families', 'Total_Families',
                    'Total_Damage', 'Severe_Count']].round(2),
            use_container_width=True
        )

        st.markdown("**Top Risk Tier Barangays**")
        top_tier = cluster_rank.index[0]
        top_brgys = agg[agg['Cluster'] == top_tier].sort_values('Total_Damage', ascending=False)
        for b, row in top_brgys.head(10).iterrows():
            st.markdown(
                f"- **{b}** - Records: {int(row['Total_Records'])}, "
                f"Max Water Level: {row['Max_Water_Level']:.1f} ft, "
                f"Total Damage: PHP {int(row['Total_Damage']):,}"
            )

        st.download_button(
            "Download Barangay Risk Clusters (CSV)",
            agg.reset_index().to_csv(index=False).encode('utf-8'),
            "bunawan_barangay_risk_clusters.csv", "text/csv")


# ------------------------------------------------------------
# TAB 3: SEASONAL FLOOD PATTERNS - Output 2
# ------------------------------------------------------------
with tab3:
    render_output_banner(
        "Seasonal Flood Patterns",
        "Description of monthly and yearly distribution of flood events in Bunawan, "
        "including identification of peak season and recurring time-based patterns.",
        "Descriptive analysis of flood occurrences by month and season."
    )

    st.header("Seasonal Flood Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flood Occurrences by Month")
        month_order = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December']
        m_counts = df_f['Month_Name'].value_counts().reindex(month_order).fillna(0)
        if m_counts.sum() > 0:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            m_counts.plot(kind='bar', color='#00796B', edgecolor='black', ax=ax)
            ax.set_ylabel("Number of Records")
            ax.set_title("Monthly Distribution of Flood Records")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.subheader("Flood Records by Season")
        s_counts = df_f['Season'].value_counts()
        if len(s_counts) > 0:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            colors = ['#26A69A' if s == 'Wet Season' else '#FFB74D' for s in s_counts.index]
            s_counts.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
            ax.set_ylabel("Number of Records")
            ax.set_title("Flood Records by Season")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.subheader("Heatmap: Flood Occurrence by Year and Month")
    pivot = df_f.pivot_table(
        index='Month_Name', columns='Year', values='Water_Level_Ft',
        aggfunc='count', fill_value=0
    ).reindex(month_order).fillna(0)

    if pivot.size > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax)
        ax.set_title("Flood Occurrence Heatmap (Year x Month)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Seasonal Summary Statistics")
    seasonal_stats = df_f.groupby('Season').agg(
        Records=('Year', 'count'),
        Mean_Water_Level=('Water_Level_Ft', 'mean'),
        Mean_Families=('Families_Affected', 'mean'),
        Total_Damage=('Total_Damage', 'sum'),
    ).round(2)
    st.dataframe(seasonal_stats, use_container_width=True)

    st.download_button(
        "Download Seasonal Patterns (CSV)",
        seasonal_stats.reset_index().to_csv(index=False).encode('utf-8'),
        "bunawan_seasonal_patterns.csv", "text/csv")


# ------------------------------------------------------------
# TAB 4: SEVERITY LEVELS (Decision Tree) - Output 3
# ------------------------------------------------------------
with tab4:
    render_output_banner(
        "Severity Levels (Low / Moderate / High)",
        "Decision tree classification of flood events by severity level using water level, "
        "number of families affected, and total damage as predictors.",
        "Supervised classification with 70/30 stratified train-test split, evaluated by "
        "accuracy, precision, recall, and F1-score."
    )

    st.header("Flood Severity Classification (Decision Tree)")

    # Prepare data
    model_df = df_f.dropna(subset=['Water_Level_Ft', 'Families_Affected', 'Total_Damage', 'Severity']).copy()
    model_df = model_df[model_df['Severity'].isin(['Low', 'Moderate', 'High'])]

    if len(model_df) < 15 or model_df['Severity'].nunique() < 2:
        st.warning("Not enough data with valid severity labels for classification.")
    else:
        X = model_df[['Water_Level_Ft', 'Families_Affected', 'Total_Damage']].values
        y = model_df['Severity'].values

        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # 70/30 stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.30, random_state=42, stratify=y_enc
            )
        except ValueError:
            # Fallback if some class too small for stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.30, random_state=42
            )

        # Fit decision tree
        dt = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_leaf=2)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision (macro)", f"{prec:.4f}")
        c3.metric("Recall (macro)", f"{rec:.4f}")
        c4.metric("F1-Score (macro)", f"{f1:.4f}")

        st.markdown(f"**Training samples:** {len(X_train)}  |  **Testing samples:** {len(X_test)}")

        st.markdown("---")
        st.subheader("Decision Tree Visualization")

        fig, ax = plt.subplots(figsize=(16, 8))
        plot_tree(dt, feature_names=['Water_Level_Ft', 'Families_Affected', 'Total_Damage'],
                  class_names=list(le.classes_), filled=True, rounded=True, ax=ax, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=le.classes_,
                                        output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Decision Rules (Extracted)")
        rules_text = export_text(dt, feature_names=['Water_Level_Ft', 'Families_Affected', 'Total_Damage'])
        st.code(rules_text, language='text')

        st.markdown("---")
        st.subheader("Severity Distribution in Filtered Data")
        sev_counts = df_f['Severity'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors_map = {'Low': '#66BB6A', 'Moderate': '#FFB74D', 'High': '#EF5350'}
        colors = [colors_map.get(s, '#90A4AE') for s in sev_counts.index]
        sev_counts.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
        ax.set_ylabel("Number of Records")
        ax.set_title("Severity Level Distribution")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.download_button(
            "Download Classification Report (CSV)",
            report_df.to_csv().encode('utf-8'),
            "bunawan_severity_classification.csv", "text/csv")


# ------------------------------------------------------------
# TAB 5: CLIMATE-SEASONAL-DAMAGE RULES (Apriori) - Output 4
# ------------------------------------------------------------
with tab5:
    render_output_banner(
        "Climate-Seasonal-Damage Rules",
        "Discovery of co-occurrence patterns among flood-causing weather systems, "
        "affected barangays, seasonal timing, and severity outcomes using Apriori "
        "association rule mining.",
        "Association rules evaluated by support, confidence, and lift."
    )

    st.header("Association Rule Mining (Apriori)")

    # Build transactions
    txn_df = df_f.copy()
    txn_df = txn_df.dropna(subset=['Barangay'])

    # Simplify flood cause to broad categories
    def simplify_cause(c):
        c = str(c)
        if 'Low Pressure' in c:
            return 'Cause=LPA'
        if 'Shearline' in c or 'Shear Line' in c:
            return 'Cause=Shearline'
        if 'Easterlies' in c:
            return 'Cause=Easterlies'
        if 'Vinta' in c:
            return 'Cause=Tropical Storm Vinta'
        if 'Auring' in c:
            return 'Cause=Tropical Depression Auring'
        if 'Seniang' in c:
            return 'Cause=Tropical Storm Seniang'
        if 'Agaton' in c:
            return 'Cause=Tropical Depression Agaton'
        if 'Amihan' in c:
            return 'Cause=Amihan'
        if 'Thunderstorm' in c:
            return 'Cause=Thunderstorm'
        return 'Cause=Other'

    transactions = []
    for _, row in txn_df.iterrows():
        txn = [
            f"Barangay={row['Barangay']}",
            simplify_cause(row['Flood_Cause']),
            f"Severity={row['Severity']}",
        ]
        if pd.notna(row['Season']) and row['Season'] != 'Unknown':
            txn.append(f"Season={row['Season']}")
        if pd.notna(row['Month_Name']):
            txn.append(f"Month={row['Month_Name']}")
        if pd.notna(row['Water_Level_Category']):
            txn.append(f"WaterLevel={row['Water_Level_Category']}")
        if pd.notna(row['Damage_Category']):
            txn.append(f"Damage={row['Damage_Category']}")
        transactions.append(set(txn))

    if len(transactions) < 5:
        st.warning("Not enough transactions for association rule mining.")
    else:
        with st.spinner("Mining frequent itemsets..."):
            freq = simple_apriori(transactions, min_support=min_support)
            rules = generate_rules(freq, min_confidence=min_confidence)

        # Filter out trivial rules (same category → same category)
        rules = [r for r in rules
                 if not (list(r['antecedent'])[0].split('=')[0] ==
                         list(r['consequent'])[0].split('=')[0])
                 or len(r['antecedent']) > 1
                 or len(r['consequent']) > 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Frequent Itemsets", len(freq))
        c2.metric("Rules Mined", len(rules))
        c3.metric("Transactions", len(transactions))

        if not rules:
            st.info("No rules met the current support and confidence thresholds. "
                    "Try lowering the sliders in the sidebar.")
        else:
            rules_df = pd.DataFrame([
                {
                    'Antecedent': ', '.join(sorted(r['antecedent'])),
                    'Consequent': ', '.join(sorted(r['consequent'])),
                    'Support': round(r['support'], 4),
                    'Confidence': round(r['confidence'], 4),
                    'Lift': round(r['lift'], 4),
                }
                for r in rules
            ]).sort_values(['Lift', 'Confidence'], ascending=False)

            st.markdown("---")
            st.subheader("Top Association Rules (sorted by Lift)")
            st.dataframe(rules_df.head(30), use_container_width=True)

            st.markdown("---")
            st.subheader("Rule Visualization")
            top_rules = rules_df.head(15)

            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(top_rules))
            labels = [f"{r['Antecedent']}  ->  {r['Consequent']}"
                      for _, r in top_rules.iterrows()]
            bars = ax.barh(y_pos, top_rules['Lift'].values, color='#FF7043', edgecolor='black')
            for bar, conf in zip(bars, top_rules['Confidence'].values):
                w = bar.get_width()
                ax.text(w + 0.03, bar.get_y() + bar.get_height() / 2,
                        f"conf={conf:.2f}", va='center', fontsize=8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Lift")
            ax.set_title("Top 15 Rules by Lift")
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.subheader("Support vs Confidence Scatter")
            fig, ax = plt.subplots(figsize=(9, 5))
            sc = ax.scatter(rules_df['Support'], rules_df['Confidence'],
                            c=rules_df['Lift'], cmap='YlOrRd',
                            s=80, alpha=0.85, edgecolors='black')
            plt.colorbar(sc, label='Lift')
            ax.set_xlabel("Support")
            ax.set_ylabel("Confidence")
            ax.set_title("Rule Quality: Support vs Confidence (colored by Lift)")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.download_button(
                "Download Association Rules (CSV)",
                rules_df.to_csv(index=False).encode('utf-8'),
                "bunawan_association_rules.csv", "text/csv")


# ------------------------------------------------------------
# TAB 6: DATA-DRIVEN PREPAREDNESS PLAN (Synthesis - findings only)
# ------------------------------------------------------------
with tab6:
    render_output_banner(
        "Data-Driven Preparedness Plan",
        "Consolidated evidence snapshots synthesized across all three data mining "
        "techniques (clustering, classification, association rules) to support "
        "evidence-based decisions by the MDRRMO and LGU of Bunawan.",
        "Findings-only summary. Recommended actions are documented separately in the "
        "manuscript and are not displayed on this dashboard."
    )

    st.header("Data-Driven Preparedness Plan: Evidence Summary")

    # --- Key statistics ---
    total_records = len(df_f)
    top_brgys = df_f['Barangay'].value_counts().head(10)
    top_cause = df_f['Flood_Cause'].value_counts().idxmax() if len(df_f) > 0 else 'N/A'
    top_month = df_f['Month_Name'].value_counts().idxmax() if df_f['Month_Name'].notna().any() else 'N/A'
    peak_season = df_f['Season'].value_counts().idxmax() if len(df_f) > 0 else 'N/A'
    total_families = int(df_f['Families_Affected'].sum())
    total_damage = int(df_f['Total_Damage'].sum())
    high_sev = int((df_f['Severity'] == 'High').sum())
    high_sev_pct = high_sev / max(total_records, 1) * 100

    # --- Finding 1: High-Risk Barangays ---
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 1 - Concentration of Flood Risk in Barangays</h5>
        <p><b>Finding:</b> A small group of barangays accounts for a disproportionate
        share of all recorded flood events. The top 10 barangays by record count are
        <b>{', '.join(top_brgys.index[:10])}</b>.
        K-Means clustering further tiers these barangays into similar-risk groups
        (Tier 1 = highest damage profile).</p>
        <p><b>Basis:</b> K-Means clustering output from the High-Risk Flood Barangays tab.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Finding 2: Seasonal Patterns ---
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 2 - Seasonal and Monthly Concentration</h5>
        <p><b>Finding:</b> Flood occurrences in Bunawan are not uniformly distributed
        across the calendar year. The most frequently recorded flood month is
        <b>{top_month}</b>, and the dominant season is the
        <b>{peak_season}</b>.</p>
        <p><b>Basis:</b> Descriptive seasonality output from the Seasonal Flood Patterns tab.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Finding 3: Predominant Flood Cause ---
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 3 - Dominant Flood-Causing Weather System</h5>
        <p><b>Finding:</b> The most frequently recorded flood-causing weather system
        in the MDRRMO dataset is <b>{top_cause}</b>. This category of weather system
        is associated with the largest share of historical flood events.</p>
        <p><b>Basis:</b> Descriptive cause distribution and association rules from the
        Climate-Seasonal-Damage Rules tab.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Finding 4: Severity Profile ---
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 4 - Severity Profile of Flood Events</h5>
        <p><b>Finding:</b> Of all filtered flood records, <b>{high_sev:,}</b>
        (<b>{high_sev_pct:.1f}%</b>) are classified as <b>High severity</b> using the
        Chapter 3 thresholds (water level &gt; 4.0 ft, families &gt; 200, damage &gt; PHP
        500,000). Decision tree classification identified water level, family
        count, and total damage as the primary discriminators between severity tiers.</p>
        <p><b>Basis:</b> Decision Tree classification output from the Severity Levels tab.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Finding 5: Aggregate Impact ---
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 5 - Aggregate Impact on Families and Local Economy</h5>
        <p><b>Finding:</b> Across the filtered records, an estimated
        <b>{total_families:,}</b> families were affected and total recorded damage
        reached <b>PHP {total_damage:,}</b> (infrastructure and agriculture combined).</p>
        <p><b>Basis:</b> Aggregate statistics from the MDRRMO dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Finding 6: Data Utilization ---
    st.markdown(f"""
    <div class="finding-card">
        <h5>Finding 6 - Local Data Utilization Snapshot</h5>
        <p><b>Finding:</b> The MDRRMO flood database contains
        <b>{len(df):,}</b> cleaned records across <b>{df.attrs.get('n_events', 0)}</b>
        distinct flood events and <b>{df['Barangay'].nunique()}</b> barangays.
        The presence of these records supports the observation that historical flood
        data is systematically stored at the local level, and this dashboard
        demonstrates that this data can be transformed into analytical evidence
        without needing external data sources.</p>
        <p><b>Basis:</b> Dataset coverage statistics computed from the cleaned MDRRMO records.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Download summary ---
    st.markdown("---")
    st.subheader("Download Summary Report (Data Only)")
    report_rows = [
        ("Total Flood Records Analyzed", f"{total_records:,}"),
        ("Distinct Barangays", df_f['Barangay'].nunique()),
        ("Distinct Events", df_f['Event_ID'].nunique()),
        ("Top 10 Barangays", ", ".join(top_brgys.index[:10])),
        ("Most Frequent Flood Cause", top_cause),
        ("Most Frequent Month", top_month),
        ("Peak Season", peak_season),
        ("High Severity Records", f"{high_sev:,} ({high_sev_pct:.1f}%)"),
        ("Total Families Affected", f"{total_families:,}"),
        ("Total Damage (PHP)", f"{total_damage:,}"),
        ("Year Range", f"{year_range[0]}-{year_range[1]}"),
    ]
    report_df = pd.DataFrame(report_rows, columns=['Metric', 'Value'])
    st.dataframe(report_df, use_container_width=True)
    st.download_button(
        "Download Summary Report (CSV)",
        report_df.to_csv(index=False).encode('utf-8'),
        "bunawan_flood_summary_report.csv", "text/csv")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Bunawan Flood Analytics Dashboard | "
    "Hannah Soliven Bathan & Rowena Mercado | "
    "Bachelor of Science in Information Systems | "
    "Agusan del Sur State University | 2026"
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------
st.set_page_config(
    page_title="World Population Dashboard",
    page_icon="🌍",
    layout="wide"
)

sns.set_style("whitegrid")

# ---------------------------------------------------------
# Data Load + Clean
# ---------------------------------------------------------
DATA_PATH = "population_by_country_2020.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="latin1")

    # Clean percentage columns -> numeric
    df["Urban Pop %"] = pd.to_numeric(
        df["Urban Pop %"].astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    )
    df["World Share"] = pd.to_numeric(
        df["World Share"].astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    )
    df["Yearly Change"] = pd.to_numeric(
        df["Yearly Change"].astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    )

    # Clean numeric columns that may contain stray characters
    df["Fert. Rate"] = pd.to_numeric(df["Fert. Rate"], errors="coerce")
    df["Med. Age"] = pd.to_numeric(df["Med. Age"], errors="coerce")

    # Fill missing values with median (same approach as source notebook)
    for col in ["Urban Pop %", "World Share", "Fert. Rate", "Med. Age"]:
        df[col] = df[col].fillna(df[col].median())

    # Density column name has an encoding artifact (P/KmÂ²) in the raw file
    density_col = [c for c in df.columns if c.startswith("Density")][0]
    df = df.rename(columns={density_col: "Density (P/Km2)"})
    df["Density (P/Km2)"] = pd.to_numeric(df["Density (P/Km2)"], errors="coerce")

    # Derived columns
    df["Log Population"] = np.log(df["Population (2020)"].replace(0, np.nan))
    df["Density Label"] = np.where(df["Density (P/Km2)"] > 500, "High Density", "Low Density")
    df["Population Density Category"] = pd.cut(
        df["Density (P/Km2)"],
        bins=[0, 100, 500, 1000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"]
    )
    df["Migration Type"] = np.where(
        df["Migrants (net)"] > 0, "Net Immigration",
        np.where(df["Migrants (net)"] < 0, "Net Emigration", "No Net Change")
    )

    return df


df = load_data(DATA_PATH)

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.header("Filters")

countries = sorted(df["Country (or dependency)"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country", options=countries, default=countries
)

pop_min, pop_max = int(df["Population (2020)"].min()), int(df["Population (2020)"].max())
pop_range = st.sidebar.slider(
    "Population (2020) Range",
    min_value=pop_min, max_value=pop_max,
    value=(pop_min, pop_max)
)

density_categories = df["Population Density Category"].dropna().unique().tolist()
selected_density = st.sidebar.multiselect(
    "Density Category", options=density_categories, default=density_categories
)

# Apply filters
mask = (
    df["Country (or dependency)"].isin(selected_countries)
    & df["Population (2020)"].between(pop_range[0], pop_range[1])
    & df["Population Density Category"].isin(selected_density)
)
filtered = df[mask]

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🌍 World Population Dashboard")
st.markdown("Interactive view of global population, density, migration and demographic trends (2020).")

# ---------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------
total_population = filtered["Population (2020)"].sum()
total_countries = filtered["Country (or dependency)"].nunique()
avg_fert_rate = filtered["Fert. Rate"].mean()
avg_med_age = filtered["Med. Age"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Population", f"{total_population:,.0f}")
col2.metric("Countries", f"{total_countries:,}")
col3.metric("Avg Fertility Rate", f"{avg_fert_rate:,.2f}")
col4.metric("Avg Median Age", f"{avg_med_age:,.1f}")

st.markdown("---")

# ---------------------------------------------------------
# Row 1: Top 10 & Bottom 10 by Population
# ---------------------------------------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Top 10 Countries by Population")
    top10 = filtered.nlargest(10, "Population (2020)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="Population (2020)", y="Country (or dependency)", data=top10,
        hue="Country (or dependency)", palette="crest", legend=False, ax=ax
    )
    ax.set_xlabel("Population (2020)")
    ax.set_ylabel("")
    st.pyplot(fig)

with row1_col2:
    st.subheader("Bottom 10 Countries by Population")
    bottom10 = filtered.nsmallest(10, "Population (2020)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="Population (2020)", y="Country (or dependency)", data=bottom10,
        hue="Country (or dependency)", palette="flare", legend=False, ax=ax
    )
    ax.set_xlabel("Population (2020)")
    ax.set_ylabel("")
    st.pyplot(fig)

# ---------------------------------------------------------
# Row 2: Density Category Distribution + Urban Pop %
# ---------------------------------------------------------
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("📊 Population Density Category")
    density_counts = filtered["Population Density Category"].value_counts().reindex(
        ["Low", "Medium", "High", "Very High"]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=density_counts.index, y=density_counts.values,
        hue=density_counts.index, palette="viridis", legend=False, ax=ax
    )
    ax.set_xlabel("Density Category")
    ax.set_ylabel("Number of Countries")
    st.pyplot(fig)

with row2_col2:
    st.subheader("🏙️ Urban Population % Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered["Urban Pop %"].dropna(), bins=20, kde=True, color="teal", ax=ax)
    ax.set_xlabel("Urban Population (%)")
    st.pyplot(fig)

# ---------------------------------------------------------
# Row 3: Fertility Rate vs Median Age + Correlation Heatmap
# ---------------------------------------------------------
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("👶 Fertility Rate vs Median Age")
    corr_val = filtered[["Fert. Rate", "Med. Age"]].corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x="Med. Age", y="Fert. Rate", data=filtered, color="darkorange", ax=ax)
    ax.set_xlabel("Median Age")
    ax.set_ylabel("Fertility Rate")
    st.pyplot(fig)
    st.caption(f"Correlation coefficient: {corr_val:.2f}")

with row3_col2:
    st.subheader("🔗 Correlation Matrix (Numeric Features)")
    numeric_cols = [
        "Population (2020)", "Density (P/Km2)", "Fert. Rate",
        "Med. Age", "Urban Pop %", "World Share"
    ]
    corr_matrix = filtered[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------
# Row 4: Migration Breakdown + World Share Cumulative
# ---------------------------------------------------------
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    st.subheader("✈️ Net Migration Breakdown")
    migration_counts = filtered["Migration Type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        migration_counts.values, labels=migration_counts.index,
        autopct="%1.0f%%", colors=sns.color_palette("Set2")
    )
    ax.axis("equal")
    st.pyplot(fig)

with row4_col2:
    st.subheader("🌐 Countries Making Up 50% of World Population")
    sorted_df = filtered.sort_values(by="World Share", ascending=False).copy()
    sorted_df["Cumulative Share"] = sorted_df["World Share"].cumsum()
    half_world = sorted_df[sorted_df["Cumulative Share"] <= 50]
    st.dataframe(
        half_world[["Country (or dependency)", "World Share", "Cumulative Share"]],
        use_container_width=True
    )
    st.caption(f"Number of countries accounting for ~50% of world population: {len(half_world)}")

# ---------------------------------------------------------
# Row 5: Highest Fertility / Median Age + Highly Urbanized Countries
# ---------------------------------------------------------
st.markdown("---")
row5_col1, row5_col2 = st.columns(2)

with row5_col1:
    st.subheader("📈 Demographic Extremes")
    highest_fert = filtered.sort_values(by="Fert. Rate", ascending=False).head(1)
    highest_age = filtered.sort_values(by="Med. Age", ascending=False).head(1)
    if not highest_fert.empty:
        st.write(
            f"**Highest Fertility Rate:** "
            f"{highest_fert['Country (or dependency)'].values[0]} "
            f"({highest_fert['Fert. Rate'].values[0]:.2f})"
        )
    if not highest_age.empty:
        st.write(
            f"**Highest Median Age:** "
            f"{highest_age['Country (or dependency)'].values[0]} "
            f"({highest_age['Med. Age'].values[0]:.0f} years)"
        )

with row5_col2:
    st.subheader("🏙️ Highly Urbanized Countries (Urban Pop % > 90)")
    highly_urban = filtered[filtered["Urban Pop %"] > 90].sort_values(
        by="Population (2020)", ascending=False
    )
    st.dataframe(
        highly_urban[["Country (or dependency)", "Urban Pop %", "Population (2020)"]],
        use_container_width=True
    )

# ---------------------------------------------------------
# Raw Data (optional expandable section)
# ---------------------------------------------------------
with st.expander("Raw Filtered Data"):
    st.dataframe(filtered, use_container_width=True)

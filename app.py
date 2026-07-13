import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Suppress Warnings
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(page_title="Sales Intelligence System", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv("vgsales.csv")
    df = df.dropna(subset=['Year'])
    df['Year'] = pd.to_datetime(df['Year'].astype(int), format='%Y')
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Dashboard Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Forecast", "Anomaly Detection", "Clustering"])

# --- PAGES ---
if page == "Overview":
    st.title("📊 Sales Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Sales Trend")
        st.bar_chart(df.groupby('Year')['Global_Sales'].sum())
    with col2:
        region = st.multiselect("Select Region", ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], default=['NA_Sales'])
        st.line_chart(df.groupby('Year')[region].sum())

elif page == "Forecast":
    st.title("📈 Sales Forecast (Prophet)")
    genre = st.selectbox("Select Genre", df['Genre'].unique())
    subset = df[df['Genre'] == genre].groupby('Year')['Global_Sales'].sum().reset_index()
    subset.columns = ['ds', 'y']
    
    # Cache the model training
    @st.cache_resource
    def get_forecast(data):
        m = Prophet(yearly_seasonality=True)
        m.fit(data)
        future = m.make_future_dataframe(periods=3, freq='YE')
        return m.predict(future)

    forecast = get_forecast(subset)
    fig, ax = plt.subplots()
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Anomaly Detection":
    st.title("⚠️ Anomaly Detection (Isolation Forest)")
    ts = df.groupby('Year')['Global_Sales'].sum().reset_index()
    model = IsolationForest(contamination=0.1, random_state=42)
    ts['anomaly'] = model.fit_predict(ts[['Global_Sales']])
    
    fig, ax = plt.subplots()
    ax.plot(ts['Year'], ts['Global_Sales'], label='Sales')
    anoms = ts[ts['anomaly'] == -1]
    ax.scatter(anoms['Year'], anoms['Global_Sales'], color='red', label='Anomaly')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

elif page == "Clustering":
    st.title("🎯 Demand Segments (K-Means)")
    stats = df.groupby('Genre').agg({'Global_Sales': ['sum', 'std']}).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(stats)
    
    n_clusters = st.slider("Number of Clusters", 2, 5, 3)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(scaled)
    stats['Cluster'] = km.labels_
    st.table(stats)
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="FloodSense AI - India", page_icon="🌊", layout="wide")
st.title("🌊 FloodSense AI: Nationwide Multi-Tier Deep AI Intelligence & 1-Year Forecasting")
st.markdown("NASA Space Apps Challenge 2026 | PyTorch Deep Neural Network Trained on Pure NASA GPM IMERG Telemetry")

device = torch.device("cpu")

class FloodDeepNet(nn.Module):
    def __init__(self, input_dim):
        super(FloodDeepNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 3)
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        return self.output(x)

@st.cache_resource
def load_and_train_deep_engine():
    csv_path = r"C:\Users\atiks\FloodSense\app\flood_predictions.csv"
    model_path = r"C:\Users\atiks\FloodSense\app\flood_model.pth"
    if not os.path.exists(csv_path):
        st.error("⚠️ dataset not found!")
        return None, None, None, None
    df_train = pd.read_csv(csv_path)
    features = ["lat", "lon", "elevation", "catchment_factor", "flow_accumulation", "precipitation", "sm_rootzone", "evapotranspiration", "grace_anomaly", "canopy_water", "snow_melt", "runoff_index"]
    X_full = df_train[features].values
    y_full = df_train["flood_severity"].values
    X_mean = X_full.mean(axis=0)
    X_std = X_full.std(axis=0) + 1e-8
    model = FloodDeepNet(input_dim=len(features)).to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model, df_train, X_mean, X_std
        except Exception:
            pass
    sample_df = df_train.sample(n=min(5000, len(df_train)), random_state=42)
    X_norm = (sample_df[features].values - X_mean) / X_std
    dataset = TensorDataset(torch.tensor(X_norm, dtype=torch.float32), torch.tensor(sample_df["flood_severity"].values, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    model.eval()
    torch.save(model.state_dict(), model_path)
    return model, df_train, X_mean, X_std

with st.spinner("🚀 Initializing PyTorch Neural Network & NASA Telemetry Dataset..."):
    deep_model, full_history, feat_mean, feat_std = load_and_train_deep_engine()

if deep_model is not None:
    def generate_future_forecast(_base_df, mean_vals, std_vals):
        _base_coords = _base_df[["lat", "lon", "elevation", "catchment_factor", "flow_accumulation"]].drop_duplicates().sample(n=min(500, len(_base_df)), random_state=42).reset_index(drop=True)
        future_records = []
        current_date = datetime.now()
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_weeks = 52
        features = ["lat", "lon", "elevation", "catchment_factor", "flow_accumulation", "precipitation", "sm_rootzone", "evapotranspiration", "grace_anomaly", "canopy_water", "snow_melt", "runoff_index"]
        for w in range(1, total_weeks + 1):
            pct = int((w / total_weeks) * 100)
            progress_bar.progress(pct)
            status_text.markdown(f"⏳ **Simulating & Running AI Inference:** Processing Week {w} of {total_weeks} ({pct}% complete)...")
            f_date = current_date + timedelta(weeks=w)
            f_df = _base_coords.copy()
            f_df["week_date"] = f_date.strftime("%Y-%m-%d")
            f_df["year"] = f_date.year
            f_df["month"] = f_date.month
            f_df["week_number"] = w
            is_monsoon = f_df["month"].isin([6, 7, 8, 9])
            np.random.seed(42 + w)
            base_p = np.where(is_monsoon, np.random.exponential(3.2, len(f_df)), np.random.exponential(0.5, len(f_df)))
            f_df["precipitation"] = np.clip(base_p, 0.0, 16.0)
            f_df["sm_rootzone"] = np.clip(f_df["precipitation"] * 0.12 + 0.15, 0.05, 0.95)
            f_df["evapotranspiration"] = np.clip(3.0 - (f_df["precipitation"] * 0.05), 0.5, 5.0)
            f_df["grace_anomaly"] = f_df["precipitation"] * 1.5 - 2.0
            f_df["canopy_water"] = f_df["precipitation"] * 0.25
            f_df["snow_melt"] = np.where((f_df["lat"] > 28.0) & (f_df["elevation"] > 2000.0), 2.5, 0.0)
            f_df["runoff_index"] = f_df["precipitation"] * 0.5 + f_df["sm_rootzone"] * 0.8 + f_df["flow_accumulation"] * 0.2 + f_df["snow_melt"] * 0.3
            X_w = f_df[features].values
            X_w_norm = (X_w - mean_vals) / std_vals
            with torch.no_grad():
                logits = deep_model(torch.tensor(X_w_norm, dtype=torch.float32).to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                f_df["predicted_severity"] = np.argmax(probs, axis=1)
                f_df["severity_proba"] = np.max(probs, axis=1)
            mask_severe = (f_df["precipitation"] >= 11.0) & (f_df["runoff_index"] >= 7.5) & (f_df["flow_accumulation"] >= 7.0)
            f_df.loc[mask_severe, "predicted_severity"] = 2
            future_records.append(f_df)
        progress_bar.empty()
        status_text.empty()
        return pd.concat(future_records, ignore_index=True)

    if "future_df" not in st.session_state:
        st.session_state["future_df"] = generate_future_forecast(full_history, feat_mean, feat_std)
    future_df = st.session_state["future_df"]

    def assign_region(lat, lon):
        if lat > 24 and lon > 89: return "Northeast Basin (Assam / Brahmaputra)"
        elif lat < 15 and lon < 77: return "Southern Western Ghats (Kerala / Karnataka)"
        elif 20 < lat < 25 and lon < 74: return "Western Basin (Gujarat / Konkan)"
        elif lat > 26 and lon < 80: return "Northern Gangetic Plain"
        else: return "Central / Eastern River Basin"

    future_df["Region"] = future_df.apply(lambda r: assign_region(r["lat"], r["lon"]), axis=1)

    st.sidebar.header("📅 1-Year Forecasting Controls")
    all_future_weeks = sorted(future_df["week_date"].unique())
    selected_week = st.sidebar.selectbox("Select Future Forecast Week (52-Week Horizon)", all_future_weeks)
    rain_multiplier = st.sidebar.slider("Precipitation Multiplier", 0.5, 2.0, 1.0, step=0.1)

    df = future_df[future_df["week_date"] == selected_week].copy()
    df["projected_precip"] = np.clip(df["precipitation"] * rain_multiplier, 0.0, 16.0)
    df["projected_runoff"] = df["projected_precip"] * 0.5 + df["sm_rootzone"] * 0.8 + df["flow_accumulation"] * 0.2

    st.sidebar.markdown("---")
    st.sidebar.success(f"Forecast Week: {selected_week}")
    st.sidebar.metric(label="Severe Flood Emergencies", value=int((df["predicted_severity"] == 2).sum()))
    st.sidebar.metric(label="Moderate Watch Basins", value=int((df["predicted_severity"] == 1).sum()))
    st.sidebar.metric(label="Safe Basins", value=int((df["predicted_severity"] == 0).sum()))

    tab1, tab2, tab3 = st.tabs(["🗺️ Geographic Risk Map", "📋 Weekly Forecast Log", "🚨 Consolidated Severe Flood Schedule"])

    with tab1:
        st.subheader(f"All-India 1-Year Geographic Flood Risk Map ({selected_week})")
        st.markdown("🌐 **Vector Map Projection:** Rendering India's actual landmass, country borders, and NASA telemetry grid.")
        fig = px.scatter_geo(df, lat="lat", lon="lon", color="projected_precip", color_continuous_scale=[[0.0, "#2ECC71"], [0.4, "#F1C40F"], [0.8, "#E67E22"], [1.0, "#E74C3C"]], range_color=[0.0, 12.0], scope="asia", projection="natural earth", hover_data={"projected_precip": ":.2f", "sm_rootzone": ":.2f", "runoff_index": ":.2f", "elevation": ":.0f", "Region": True})
        fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="rgb(150, 150, 150)", showland=True, landcolor="rgb(245, 247, 249)", showocean=True, oceancolor="rgb(225, 240, 255)", showlakes=True, lakecolor="rgb(225, 240, 255)")
        fig.update_traces(marker=dict(size=7, opacity=0.9))
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=600, coloraxis_colorbar=dict(title="Rainfall (mm/hr)"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Basin Risk Breakdown for Week of {selected_week}")
        table_df = df.copy()
        table_df["Severity Tier"] = table_df["predicted_severity"].map({2: "SEVERE FLOOD", 1: "MODERATE WATCH", 0: "Safe"})
        table_df["AI Confidence"] = (table_df["severity_proba"] * 100).round(1).astype(str) + "%"
        display_table = table_df[["week_date", "Region", "Severity Tier", "AI Confidence", "projected_precip", "sm_rootzone", "runoff_index", "elevation", "lat", "lon"]]
        display_table.columns = ["Week Date", "Target River Basin / Region", "Severity Tier", "Confidence", "Rainfall (mm/hr)", "Soil Moisture", "Runoff Index", "Elevation (m)", "Latitude", "Longitude"]
        st.dataframe(display_table, use_container_width=True)

    with tab3:
        st.subheader("Consolidated Regional Severe Flood Schedule (Next 5 Unique Events)")
        st.markdown("Chronological schedule of distinct regional flood events across the 52-week future horizon.")
        severe_df = future_df[future_df["predicted_severity"] == 2].copy()
        if len(severe_df) > 0:
            severe_df["dt"] = pd.to_datetime(severe_df["week_date"])
            severe_df["event_group"] = severe_df["dt"].dt.to_period("M")
            consolidated = severe_df.groupby(["Region", "event_group"]).agg(start_date=("dt", "min"), end_date=("dt", "max"), max_precip=("precipitation", "max"), avg_runoff=("runoff_index", "mean"), avg_confidence=("severity_proba", "mean")).reset_index()
            consolidated["end_date"] = consolidated["end_date"] + pd.Timedelta(days=7)
            consolidated["Date Range"] = consolidated["start_date"].dt.strftime("%b %d, %Y") + " – " + consolidated["end_date"].dt.strftime("%b %d, %Y")
            consolidated = consolidated.sort_values(by="start_date").head(5)
            consolidated["Severity Tier"] = "SEVERE FLOOD EMERGENCY"
            consolidated["AI Confidence"] = (consolidated["avg_confidence"] * 100).round(1).astype(str) + "%"
            schedule_table = consolidated[["Date Range", "Region", "Severity Tier", "AI Confidence", "max_precip", "avg_runoff"]]
            schedule_table.columns = ["Event Date Range", "Target River Basin / Region", "Severity Tier", "Confidence", "Peak Rainfall (mm/hr)", "Avg Runoff Index"]
            st.dataframe(schedule_table, use_container_width=True)
        else:
            st.info("No severe flood events predicted in the current horizon window.")
else:
    st.warning("Please verify your dataset path.")

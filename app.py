"""
KSP Crime Intelligence & Visualization Platform
Streamlit app — reads pre-computed artifacts exported from the Kaggle training notebook.
Run: streamlit run app.py
Expected folder layout:
    app.py
    artifacts/
        crime_incidents.csv
        district_daily_risk.csv
        districts_reference.csv
        graph_edges.csv
        graph_nodes.csv
        hotspot_clusters.csv
        isolation_forest.pkl
        risk_model.pkl
        top_repeat_offenders.csv
        emerging_trend_alerts.csv
        association_rules.csv
"""

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="KSP Crime Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
)

# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------

@st.cache_data
def load_incidents():
    df = pd.read_csv(ART_DIR / "crime_incidents.csv", parse_dates=["date"])
    return df


@st.cache_data
def load_daily_risk():
    return pd.read_csv(ART_DIR / "district_daily_risk.csv", parse_dates=["date"])


@st.cache_data
def load_districts():
    return pd.read_csv(ART_DIR / "districts_reference.csv")


@st.cache_data
def load_hotspots():
    return pd.read_csv(ART_DIR / "hotspot_clusters.csv")


@st.cache_data
def load_graph_tables():
    edges = pd.read_csv(ART_DIR / "graph_edges.csv")
    nodes = pd.read_csv(ART_DIR / "graph_nodes.csv")
    return edges, nodes


@st.cache_data
def load_top_offenders():
    return pd.read_csv(ART_DIR / "top_repeat_offenders.csv")


@st.cache_data
def load_emerging_trends():
    file_path = ART_DIR / "emerging_trend_alerts.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


@st.cache_data
def load_association_rules():
    file_path = ART_DIR / "association_rules.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


@st.cache_resource
def load_risk_model():
    with open(ART_DIR / "risk_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_anomaly_model():
    with open(ART_DIR / "isolation_forest.pkl", "rb") as f:
        return pickle.load(f)


if not ART_DIR.exists():
    st.error(
        "Artifacts folder not found. Run the Kaggle training notebook, download its "
        "`artifacts/` output, and place it next to this app.py file."
    )
    st.stop()

# Load all data
incidents = load_incidents()
daily_risk = load_daily_risk()
districts = load_districts()
hotspots = load_hotspots()
edges, nodes = load_graph_tables()
top_offenders = load_top_offenders()
emerging_alerts = load_emerging_trends()
assoc_rules = load_association_rules()

# ----------------------------------------------------------------------
# Sidebar navigation + global filters
# ----------------------------------------------------------------------

st.sidebar.title("🛡️ KSP Crime Intelligence")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Geospatial Hotspot Map",
        "District Drill-down",
        "Network & Link Analysis",
        "Predictive Risk Dashboard",
        "Anomaly Detection",
        "Trend & Pattern Discovery",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
district_filter = st.sidebar.multiselect(
    "District", sorted(incidents["district"].unique()), default=[]
)
crime_filter = st.sidebar.multiselect(
    "Crime type", sorted(incidents["crime_type"].unique()), default=[]
)
date_range = st.sidebar.date_input(
    "Date range",
    value=(incidents["date"].min().date(), incidents["date"].max().date()),
)

filtered = incidents.copy()
if district_filter:
    filtered = filtered[filtered["district"].isin(district_filter)]
if crime_filter:
    filtered = filtered[filtered["crime_type"].isin(crime_filter)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(filtered):,} incidents match current filters")
st.sidebar.caption("⚠️ Demo built on synthetic data modeled on public crime taxonomies — not real KSP records.")

# ----------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------

if page == "Overview":
    st.title("AI-Driven Crime Analytics & Visualization Platform")
    st.caption("State Crime Records Bureau — Strategic Intelligence Hub (demo)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents", f"{len(filtered):,}")
    c2.metric("Resolution rate", f"{filtered['resolved'].mean()*100:.1f}%")
    c3.metric("Avg severity", f"{filtered['severity_score'].mean():.2f} / 10")
    c4.metric("Anomalous incidents", f"{filtered['is_anomaly'].sum():,}")

    col1, col2 = st.columns(2)
    with col1:
        by_crime = filtered["crime_type"].value_counts().reset_index()
        by_crime.columns = ["crime_type", "count"]
        fig = px.bar(by_crime, x="count", y="crime_type", orientation="h",
                     title="Incidents by crime type", color="count",
                     color_continuous_scale="Reds")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')
    with col2:
        by_district = filtered["district"].value_counts().head(12).reset_index()
        by_district.columns = ["district", "count"]
        fig = px.bar(by_district, x="count", y="district", orientation="h",
                     title="Top districts by incident count", color="count",
                     color_continuous_scale="Blues")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

    monthly = filtered.groupby(filtered["date"].dt.to_period("M")).size().reset_index(name="count")
    monthly["date"] = monthly["date"].dt.to_timestamp()
    fig = px.line(monthly, x="date", y="count", title="Incident trend over time", markers=True)
    st.plotly_chart(fig, width='stretch')

# ----------------------------------------------------------------------
# PAGE: Geospatial Hotspot Map
# ----------------------------------------------------------------------

elif page == "Geospatial Hotspot Map":
    st.title("Geospatial Crime Hotspot Map")
    st.caption("Replaces static Excel sheets with an interactive, spatial view of crime clusters.")

    map_mode = st.radio("Map layer", ["Individual incidents", "Hotspot clusters"], horizontal=True)

    if map_mode == "Individual incidents":
        sample = filtered.sample(min(4000, len(filtered)), random_state=1) if len(filtered) > 4000 else filtered
        fig = px.scatter_map(
            sample, lat="latitude", lon="longitude", color="crime_type",
            hover_data=["district", "date", "severity_score"],
            zoom=6, height=650, map_style="carto-positron",
        )
        st.plotly_chart(fig, width='stretch')
        if len(filtered) > 4000:
            st.caption("Showing a 4,000-point sample for render performance.")
    else:
        fig = px.scatter_map(
            hotspots, lat="center_lat", lon="center_lon", size="size",
            color="dominant_crime", hover_data=["avg_hour", "size"],
            zoom=6, height=650, map_style="carto-positron",
            title="Spatiotemporal hotspot clusters (DBSCAN)",
        )
        st.plotly_chart(fig, width='stretch')
        st.dataframe(hotspots.sort_values("size", ascending=False), width='stretch')

# ----------------------------------------------------------------------
# PAGE: District Drill-down
# ----------------------------------------------------------------------

elif page == "District Drill-down":
    st.title("District-Level Drill-down")
    dist = st.selectbox("Select district", sorted(incidents["district"].unique()))
    d_data = incidents[incidents["district"] == dist]

    c1, c2, c3 = st.columns(3)
    c1.metric("Incidents", f"{len(d_data):,}")
    c2.metric("Resolution rate", f"{d_data['resolved'].mean()*100:.1f}%")
    c3.metric("Avg severity", f"{d_data['severity_score'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(d_data, names="crime_type", title=f"Crime mix — {dist}", hole=0.4)
        st.plotly_chart(fig, width='stretch')
    with col2:
        heat = d_data.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
        fig = px.density_heatmap(heat, x="hour", y="day_of_week", z="count",
                                  title="When crime happens (hour vs weekday)", color_continuous_scale="Inferno")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Police-station breakdown")
    ps = d_data["police_station"].value_counts().reset_index()
    ps.columns = ["police_station", "incidents"]
    st.dataframe(ps, width='stretch')

# ----------------------------------------------------------------------
# PAGE: Network & Link Analysis
# ----------------------------------------------------------------------

elif page == "Network & Link Analysis":
    st.title("Criminological Network & Link Analysis")
    st.caption("Visually connects suspects, victims, and locations to reveal organized crime structures.")

    st.subheader("Top repeat / highly-connected suspects")
    st.dataframe(top_offenders, width='stretch')

    st.subheader("Explore a suspect's network")
    suspect_id = st.selectbox("Suspect ID", top_offenders["suspect_id"].tolist())

    # build ego graph (suspect + direct connections) from edge list
    sub_edges = edges[(edges["source"] == suspect_id) | (edges["target"] == suspect_id)]
    ego_nodes = set(sub_edges["source"]).union(set(sub_edges["target"]))
    node_type = dict(zip(nodes["node"], nodes["type"]))

    G = nx.Graph()
    for _, row in sub_edges.iterrows():
        G.add_edge(row["source"], row["target"])
    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), mode="lines", hoverinfo="none")

    color_map = {"suspect": "#d62728", "victim": "#1f77b4", "location": "#2ca02c"}
    node_x, node_y, node_color, node_text = [], [], [], []
    for n in G.nodes():
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        t = node_type.get(n, "unknown")
        node_color.append(color_map.get(t, "#888"))
        node_text.append(f"{n} ({t})")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
        marker=dict(size=14, color=node_color, line=dict(width=1, color="white")), hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, height=600, title=f"Ego network — {suspect_id}",
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width='stretch')
    st.markdown("🔴 Suspect &nbsp; 🔵 Victim &nbsp; 🟢 Location (police station)")

    st.subheader("Related incidents")
    st.dataframe(
        incidents[incidents["suspect_id"] == suspect_id][
            ["incident_id", "date", "district", "crime_type", "mo", "severity_score"]
        ],
        width='stretch',
    )
    
    st.markdown("---")
    st.subheader("Association Detection (Market Basket Analysis)")
    st.caption("Identifies hidden criminal associations between specific crime typologies and legal Acts invoked.")
    
    if not assoc_rules.empty:
        display_rules = assoc_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        display_rules[['support', 'confidence', 'lift']] = display_rules[['support', 'confidence', 'lift']].round(3)
        st.dataframe(display_rules.sort_values("lift", ascending=False), width='stretch')
    else:
        st.info("No association rules found. Run the latest Kaggle notebook to generate these artifacts.")

# ----------------------------------------------------------------------
# PAGE: Predictive Risk Dashboard
# ----------------------------------------------------------------------

elif page == "Predictive Risk Dashboard":
    st.title("Predictive Risk Scoring")
    st.caption("AI-driven forecast of which district-days are likely to be high-risk.")

    risk_bundle = load_risk_model()
    model = risk_bundle["model"]
    le_district = risk_bundle["district_encoder"]
    feature_cols = risk_bundle["features"]

    latest = daily_risk.sort_values("date").groupby("district").tail(1).copy()
    latest["district_enc"] = le_district.transform(latest["district"])
    X_latest = latest[feature_cols].fillna(0)
    latest["risk_probability"] = model.predict_proba(X_latest)[:, 1]
    latest = latest.sort_values("risk_probability", ascending=False)

    fig = px.bar(
        latest, x="risk_probability", y="district", orientation="h",
        color="risk_probability", color_continuous_scale="RdYlGn_r",
        title="Predicted high-risk probability by district (most recent snapshot)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Feature importance")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig2 = px.bar(importances, orientation="h", title="What drives the risk score")
    st.plotly_chart(fig2, width='stretch')

    st.subheader("Socio-economic correlation")
    merged = daily_risk.groupby("district").agg(
        incidents=("incident_count", "sum"), urbanization=("urbanization", "first"),
        unemployment=("unemployment", "first"), literacy=("literacy", "first"),
    ).reset_index()
    x_axis = st.selectbox("Socio-economic factor", ["urbanization", "unemployment", "literacy"])
    fig3 = px.scatter(merged, x=x_axis, y="incidents", text="district", trendline="ols",
                       title=f"Total incidents vs {x_axis}")
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, width='stretch')

# ----------------------------------------------------------------------
# PAGE: Anomaly Detection
# ----------------------------------------------------------------------

elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.caption("Incidents that deviate from standard behavioral patterns — for linking complex or unusual cases.")

    anomalies = filtered[filtered["is_anomaly"]]
    st.metric("Anomalous incidents in current filter", f"{len(anomalies):,}")

    fig = px.scatter_map(
        anomalies, lat="latitude", lon="longitude", color="crime_type",
        size="severity_score", hover_data=["district", "date", "hour"],
        zoom=6, height=600, map_style="carto-positron", title="Anomalous incidents map",
    )
    st.plotly_chart(fig, width='stretch')

    st.dataframe(
        anomalies[["incident_id", "date", "district", "crime_type", "hour", "severity_score", "mo"]]
        .sort_values("severity_score", ascending=False),
        width='stretch',
    )

# ----------------------------------------------------------------------
# PAGE: Trend & Pattern Discovery
# ----------------------------------------------------------------------

elif page == "Trend & Pattern Discovery":
    st.title("Pattern & Trend Discovery")

    col1, col2 = st.columns(2)
    with col1:
        crime_sel = st.selectbox("Crime type", sorted(incidents["crime_type"].unique()))
        trend = incidents[incidents["crime_type"] == crime_sel].groupby(
            incidents["date"].dt.to_period("M")
        ).size().reset_index(name="count")
        trend["date"] = trend["date"].dt.to_timestamp()
        fig = px.line(trend, x="date", y="count", title=f"{crime_sel} trend over time", markers=True)
        st.plotly_chart(fig, width='stretch')
    with col2:
        mo_counts = incidents[incidents["crime_type"] == crime_sel]["mo"].value_counts().reset_index()
        mo_counts.columns = ["modus_operandi", "count"]
        fig = px.bar(mo_counts, x="count", y="modus_operandi", orientation="h",
                     title=f"Modus Operandi breakdown — {crime_sel}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Emerging Trend Alerts (Z-Score Spikes)")
    st.caption("Visual indicators when a specific crime category spikes in a region compared to historical averages (Z-Score > 2.0).")
    
    if not emerging_alerts.empty:
        st.error(f"🚨 Detected {len(emerging_alerts)} critical spikes across the state.")
        display_alerts = emerging_alerts[['district', 'crime_type', 'year_month', 'incident_count', 'mean', 'z_score']].copy()
        display_alerts.rename(columns={'mean': 'historical_avg'}, inplace=True)
        display_alerts[['historical_avg', 'z_score']] = display_alerts[['historical_avg', 'z_score']].round(2)
        
        # FIX: Removed style.background_gradient to avoid matplotlib dependency crash
        st.dataframe(display_alerts, width='stretch')
    else:
        st.success("No emerging trends detected above the historical baseline at this time.")
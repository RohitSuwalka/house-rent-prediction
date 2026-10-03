"""
streamlit_app.py  —  House Rent Prediction Frontend
----------------------------------------------------
Run with:  streamlit run streamlit_app.py
Requires Flask API to be running at http://127.0.0.1:5000
(or adjust API_URL below to call the model directly without Flask)
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model directly (no Flask dependency needed for Streamlit) ─────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "rent_model.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "model", "encoders.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "..", "House_Rent_Dataset.csv")

@st.cache_resource
def load_artefacts():
    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH)
    return model, encoders

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model, encoders = load_artefacts()
FEATURE_COLS  = encoders["feature_cols"]
CAT_COLS      = encoders["cat_cols"]
UNIQUE_VALUES = encoders["unique_values"]

df = load_data()

# ── Helper: floor parser ──────────────────────────────────────────────────────
def parse_floor(floor_str: str):
    floor_str = str(floor_str).lower().strip()
    current = 0 if "ground" in floor_str else int((re.match(r"(\d+)", floor_str) or re.match(r"(0)", "0")).group(1))
    m_total = re.search(r"out of\s+(\d+)", floor_str)
    total   = int(m_total.group(1)) if m_total else current
    return current, total

def predict_rent(inputs: dict) -> float:
    floor_current, floor_total = parse_floor(inputs.get("floor", "1 out of 2"))
    encoded_cats = {}
    for col in CAT_COLS:
        le    = encoders[col]
        value = inputs.get(col, le.classes_[0])
        if value not in le.classes_:
            value = le.classes_[0]
        encoded_cats[col + "_enc"] = int(le.transform([value])[0])

    row = {
        "BHK":          int(inputs["BHK"]),
        "Size":         float(inputs["Size"]),
        "Bathroom":     int(inputs["Bathroom"]),
        "Floor_Current": floor_current,
        "Floor_Total":   floor_total,
        **encoded_cats,
    }
    X = np.array([[row[f] for f in FEATURE_COLS]])
    return max(0.0, float(model.predict(X)[0]))

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem; font-weight: 800; color: #1f2328;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem; color: #57606a;
        text-align: center; margin-bottom: 1.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 16px; text-align: center;
        color: white; margin: 1rem 0;
    }
    .pred-label { font-size: 1rem; opacity: 0.85; margin-bottom: 0.3rem; }
    .pred-value { font-size: 3rem; font-weight: 800; }
    .metric-card {
        background: #f7f8fa; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .metric-label { font-size: 0.85rem; color: #57606a; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #1f2328; }
    div[data-testid="stSidebar"] { background: #f7f8fa; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏠 House Rent Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered rent estimation across Indian cities · Powered by Gradient Boosting</div>', unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔧 Property Details")
    st.caption("Fill in the details below to estimate rent.")

    city               = st.selectbox("🏙️ City",              UNIQUE_VALUES["City"])
    bhk                = st.slider("🛏️ BHK",                  1, 6, 2)
    size               = st.number_input("📐 Size (sq ft)",   min_value=100, max_value=10000, value=800, step=50)
    bathroom           = st.slider("🚿 Bathrooms",            1, 6, 1)

    st.markdown("**🏢 Floor**")
    floor_current_in   = st.number_input("Current Floor",     min_value=0, max_value=50, value=1)
    floor_total_in     = st.number_input("Total Floors",      min_value=1, max_value=50, value=3)
    floor_str          = f"{floor_current_in} out of {floor_total_in}" if floor_current_in > 0 else f"Ground out of {floor_total_in}"

    area_type          = st.selectbox("📏 Area Type",          UNIQUE_VALUES["Area Type"])
    furnishing         = st.selectbox("🛋️ Furnishing Status", UNIQUE_VALUES["Furnishing Status"])
    tenant             = st.selectbox("👥 Tenant Preferred",   UNIQUE_VALUES["Tenant Preferred"])
    contact            = st.selectbox("📞 Point of Contact",   UNIQUE_VALUES["Point of Contact"])

    predict_btn = st.button("🔮 Predict Rent", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Insights", "📈 Feature Importance"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Prediction
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.subheader("Property Summary")
        summary_data = {
            "Property": ["City", "BHK", "Size", "Bathrooms", "Floor", "Area Type", "Furnishing", "Tenant", "Contact"],
            "Value":    [city, f"{bhk} BHK", f"{size} sq ft", bathroom, floor_str,
                         area_type, furnishing, tenant, contact],
        }
        st.table(pd.DataFrame(summary_data))

    with col_right:
        inputs = {
            "BHK": bhk, "Size": size, "Bathroom": bathroom,
            "floor": floor_str,
            "Area Type": area_type, "City": city,
            "Furnishing Status": furnishing,
            "Tenant Preferred": tenant,
            "Point of Contact": contact,
        }

        if predict_btn or True:   # always show a live preview
            predicted = predict_rent(inputs)
            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-label">Estimated Monthly Rent</div>
                <div class="pred-value">₹{predicted:,.0f}</div>
                <div style="opacity:0.7;font-size:0.85rem;margin-top:0.5rem">
                    {city} · {bhk} BHK · {size} sq ft · {furnishing}
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Comparable properties ──────────────────────────────────────
            st.subheader("📌 Similar Listings in Dataset")
            mask = (
                (df["City"] == city) &
                (df["BHK"] == bhk) &
                (df["Furnishing Status"] == furnishing)
            )
            similar = df[mask][["BHK", "Size", "Rent", "Floor", "Area Type",
                                  "Furnishing Status", "City"]].head(8)
            if similar.empty:
                st.info("No exact matches found — try adjusting City or BHK.")
            else:
                st.dataframe(similar.reset_index(drop=True), use_container_width=True)

    # ── Rent range gauge ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("📉 Rent Distribution for Selected City")
    city_rents = df[df["City"] == city]["Rent"]
    if not city_rents.empty:
        fig_dist = px.histogram(
            city_rents, nbins=40, color_discrete_sequence=["#667eea"],
            labels={"value": "Rent (₹)", "count": "Listings"},
            template="plotly_white",
        )
        fig_dist.add_vline(
            x=predicted, line_dash="dash", line_color="#764ba2",
            annotation_text=f"Your estimate: ₹{predicted:,.0f}",
            annotation_position="top right",
        )
        fig_dist.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Data Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Listings</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Rent</div><div class="metric-value">₹{df["Rent"].mean():,.0f}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Median Rent</div><div class="metric-value">₹{df["Rent"].median():,.0f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Cities Covered</div><div class="metric-value">{df["City"].nunique()}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        # Avg rent by city
        city_avg = df.groupby("City")["Rent"].median().sort_values(ascending=True).reset_index()
        fig_city = px.bar(
            city_avg, x="Rent", y="City", orientation="h",
            color="Rent", color_continuous_scale="Purples",
            labels={"Rent": "Median Rent (₹)"}, template="plotly_white",
            title="Median Rent by City",
        )
        fig_city.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_city, use_container_width=True)

    with c2:
        # Rent by furnishing status
        furn_avg = df.groupby("Furnishing Status")["Rent"].median().reset_index()
        fig_furn = px.bar(
            furn_avg, x="Furnishing Status", y="Rent",
            color="Furnishing Status",
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
            labels={"Rent": "Median Rent (₹)"}, template="plotly_white",
            title="Median Rent by Furnishing Status",
        )
        fig_furn.update_layout(height=340, showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_furn, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Size vs Rent scatter
        df_sample = df.sample(min(1000, len(df)), random_state=42)
        fig_sc = px.scatter(
            df_sample, x="Size", y="Rent", color="City",
            opacity=0.6, template="plotly_white",
            labels={"Size": "Size (sq ft)", "Rent": "Rent (₹)"},
            title="Size vs Rent",
        )
        fig_sc.update_layout(height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    with c4:
        # BHK distribution
        bhk_counts = df["BHK"].value_counts().reset_index()
        bhk_counts.columns = ["BHK", "Count"]
        fig_bhk = px.pie(
            bhk_counts, names="BHK", values="Count",
            color_discrete_sequence=px.colors.sequential.Purples_r,
            title="BHK Distribution", template="plotly_white",
            hole=0.4,
        )
        fig_bhk.update_layout(height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig_bhk, use_container_width=True)

    # Rent by BHK box plot
    fig_box = px.box(
        df[df["Rent"] < df["Rent"].quantile(0.98)],
        x="BHK", y="Rent", color="City",
        template="plotly_white", title="Rent Distribution by BHK across Cities",
        labels={"Rent": "Rent (₹)", "BHK": "BHK"},
    )
    fig_box.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📈 Feature Importance — What Drives Rent?")

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": importance,
    }).sort_values("Importance", ascending=True)

    # Clean display names
    name_map = {
        "BHK": "BHK", "Size": "Size (sq ft)", "Bathroom": "Bathrooms",
        "Floor_Current": "Current Floor", "Floor_Total": "Total Floors",
        "Area Type_enc": "Area Type", "City_enc": "City",
        "Furnishing Status_enc": "Furnishing Status",
        "Tenant Preferred_enc": "Tenant Preferred",
        "Point of Contact_enc": "Point of Contact",
    }
    feat_df["Feature_Label"] = feat_df["Feature"].map(name_map).fillna(feat_df["Feature"])

    fig_fi = px.bar(
        feat_df, x="Importance", y="Feature_Label", orientation="h",
        color="Importance", color_continuous_scale="Purples",
        template="plotly_white",
        labels={"Importance": "Importance Score", "Feature_Label": "Feature"},
        title="Gradient Boosting — Feature Importance",
    )
    fig_fi.update_layout(height=440, coloraxis_showscale=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info(
        "**Interpretation:** Features with higher scores have a stronger influence on the predicted rent. "
        "City, Size, and BHK typically dominate rent predictions."
    )

    # Model metrics card
    st.subheader("🎯 Model Performance")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Algorithm", "Gradient Boosting")
    with m2:
        st.metric("Estimators", "300")
    with m3:
        st.metric("Learning Rate", "0.08")

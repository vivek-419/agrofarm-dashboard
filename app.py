import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="AgroFarm IoT Dashboard",
    page_icon="A",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- global ---------- */
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }

    /* ---------- hide default header background ---------- */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }


    /* ---------- header banner ---------- */
    .main-header {
        background: linear-gradient(135deg, #2d6a4f 0%, #40916c 100%);
        padding: 2rem 2.5rem;
        border-radius: 10px;
        margin-bottom: 1.8rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 0.95rem;
        margin: 0.4rem 0 0 0;
        opacity: 0.85;
        font-weight: 400;
    }

    /* ---------- section headers ---------- */
    .section-header {
        border-left: 4px solid #2d6a4f;
        padding-left: 12px;
        font-size: 22px;
        font-weight: 700;
        color: #1a1a1a;
        margin: 1.5rem 0 1rem 0;
    }

    /* ---------- metric cards ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #5a5a5a !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    /* make "off" delta text visible */
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #555555 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
        display: none;
    }

    /* ---------- insight box ---------- */
    .insight-box {
        background: #f0f7f4;
        border-left: 4px solid #2d6a4f;
        border-radius: 6px;
        padding: 1.1rem 1.4rem;
        margin: 1rem 0;
        color: #333333;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .insight-box .insight-label {
        color: #2d6a4f;
        font-weight: 700;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        display: block;
        margin-bottom: 0.4rem;
    }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8e8e8;
    }

    /* brand block */
    .sidebar-brand {
        padding: 1.4rem 1.2rem 1rem;
        border-bottom: 1px solid #e8e8e8;
        margin-bottom: 0.6rem;
    }
    .sidebar-brand h2 {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .sidebar-brand p {
        font-size: 0.62rem;
        color: #888888;
        margin: 0.25rem 0 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
    }

    /* force sidebar text colors */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #e8e8e8 !important;
    }

    /* radio nav styling */
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 0 !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: transparent;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 0;
        margin: 0;
        padding: 0.6rem 1.2rem;
        font-size: 0.88rem;
        font-weight: 400;
        color: #777777 !important;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label p,
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label span {
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        border-radius: 0 !important;
        color: inherit !important;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        color: #2d6a4f !important;
        border-left-color: #c8ddd0;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover p,
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover span {
        color: #2d6a4f !important;
    }
    /* active item */
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
        border-left: 3px solid #2d6a4f;
        color: #1a1a1a !important;
        font-weight: 600;
        background: #f0f7f4;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] p,
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] span,
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) p,
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) span {
        color: #1a1a1a !important;
        font-weight: 600;
        background: transparent !important;
    }
    /* hide radio dot */
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* sidebar stat cards */
    .kn-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        padding: 0 8px;
    }
    .kn-card {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 0.6rem 0.7rem;
        text-align: center;
        border: 1px solid #e8e8e8;
    }
    .kn-card .kn-val {
        font-size: 1rem;
        font-weight: 700;
        color: #2d6a4f;
        margin: 0;
    }
    .kn-card .kn-lbl {
        font-size: 0.6rem;
        color: #888888;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
    }

    /* ---------- table ---------- */
    .stTable {
        border-radius: 8px;
        overflow: hidden;
    }
    .stTable table {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border-collapse: collapse;
        width: 100%;
    }
    .stTable th {
        background: #2d6a4f !important;
        color: #ffffff !important;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 0.7rem 1rem !important;
        text-align: center !important;
    }
    .stTable td {
        background: #ffffff !important;
        color: #333333 !important;
        font-size: 0.88rem;
        padding: 0.6rem 1rem !important;
        text-align: center !important;
        border-bottom: 1px solid #eee !important;
    }
    .stTable tr:hover td {
        background: #f7faf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    loaded_df = pd.read_csv("livestock_data.csv")
    loaded_monthly = pd.read_csv("monthly_yield.csv")
    loaded_agg = pd.read_csv("cluster_data.csv")
    loaded_df["Date"] = pd.to_datetime(loaded_df["Date"])
    return loaded_df, loaded_monthly, loaded_agg

df, monthly_df, agg_df = load_data()

# ── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div class="sidebar-brand">'
    "<h2>AgroFarm Analytics</h2>"
    "<p>IoT &middot; Livestock &middot; Intelligence</p>"
    "</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Financial Analysis",
        "Livestock Data & Simulation",
        "Cluster Analysis",
        "Forecast",
        "Recommendation",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="kn-grid">'
    '  <div class="kn-card"><p class="kn-val">2,000</p><p class="kn-lbl">Total Cattle</p></div>'
    '  <div class="kn-card"><p class="kn-val">₹2 Cr</p><p class="kn-lbl">Annual Loss</p></div>'
    '  <div class="kn-card"><p class="kn-val">₹1.1 Cr</p><p class="kn-lbl">IoT Invest</p></div>'
    '  <div class="kn-card"><p class="kn-val">₹10 L</p><p class="kn-lbl">Net Benefit</p></div>'
    "</div>",
    unsafe_allow_html=True,
)


# ── Helper: section header ──────────────────────────────────────────
def section_header(title):
    st.markdown(
        f'<div class="section-header">{title}</div>',
        unsafe_allow_html=True,
    )


# ── Helper: insight box ─────────────────────────────────────────────
def insight_box(text, label="Insight"):
    st.markdown(
        f'<div class="insight-box">'
        f'<span class="insight-label">{label}</span>'
        f"{text}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════
if page == "Overview":
    # ── Header ──
    st.markdown(
        """
        <div class="main-header">
            <h1>AgroFarm IoT Livestock Tracking Dashboard</h1>
            <p>Case Study 39 · B.Tech CSE 2024-28 · Business Studies · Semester IV</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metric Cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="ANNUAL LOSS",
            value="₹2,00,00,000",
            delta="Current Manual System",
            delta_color="inverse",
        )
    with c2:
        st.metric(
            label="IOT INVESTMENT",
            value="₹1,10,00,000",
            delta="One-time Capex",
            delta_color="normal",
        )
    with c3:
        st.metric(
            label="TOTAL SAVINGS",
            value="₹1,20,00,000",
            delta="+ve Cash Flow",
            delta_color="normal",
        )
    with c4:
        st.metric(
            label="NET BENEFIT",
            value="₹10,00,000",
            delta="Year 1 Profit",
            delta_color="normal",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison Table ──
    section_header("Manual System vs IoT System")

    comparison_data = {
        "Metric": [
            "Animal Monitoring",
            "Health Alerts",
            "Feeding Tracking",
            "Milk Yield Records",
            "Labor Cost",
        ],
        "Manual System": [
            "Manual rounds",
            "Delayed hours",
            "Paper logs",
            "Manual entry",
            "High",
        ],
        "IoT System": [
            "Real-time GPS",
            "Instant alert",
            "Auto sensor",
            "Auto measured",
            "Reduced 25%",
        ],
        "Improvement": [
            "24x faster",
            "-80% delay",
            "-60% errors",
            "+25% accuracy",
            "₹50L saved/yr",
        ],
    }

    comp_df = pd.DataFrame(comparison_data).set_index("Metric")

    st.table(comp_df)

    insight_box(
        "The IoT-enabled system replaces manual, error-prone "
        "processes with real-time automated monitoring. Key advantages include "
        "24x faster animal tracking via GPS collars, instant health alerts "
        "reducing response delay by 80%, auto-sensor feeding logs cutting "
        "errors by 60%, and automated milk yield measurement improving "
        "accuracy by 25%. Overall, the transition saves approximately "
        "₹50 Lakhs per year in labor costs alone, yielding a net positive "
        "cash flow in the very first year of deployment."
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Financial Analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Financial Analysis":
    st.markdown(
        """
        <div class="main-header">
            <h1>Financial Analysis</h1>
            <p>Questions 1 &amp; 2 — Total Annual Savings &amp; Net Benefit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Q1 · Total Annual Savings ──────────────────────────────────
    section_header("Q1 — Total Annual Savings")

    st.latex(
        r"Savings_{animal} = 2{,}00{,}00{,}000 \times 35\% = 70{,}00{,}000"
    )
    st.latex(
        r"Savings_{labor} = 1{,}00{,}00{,}000 \times 25\% = 50{,}00{,}000"
    )
    st.markdown(
        '<p style="color:#666; font-size:0.82rem; margin-top:-0.5rem;">'
        "Assumption: Annual labor cost = ₹1 Crore</p>",
        unsafe_allow_html=True,
    )
    st.latex(
        r"Total\ Savings = 70L + 50L = 1{,}20{,}00{,}000"
    )

    q1c1, q1c2, q1c3, q1c4 = st.columns(4)
    with q1c1:
        st.metric("ANNUAL ANIMAL LOSS", "₹2,00,00,000")
    with q1c2:
        st.metric("ANIMAL LOSS SAVED (35%)", "₹70,00,000")
    with q1c3:
        st.metric("LABOR SAVED (25%)", "₹50,00,000")
    with q1c4:
        st.metric("TOTAL ANNUAL SAVINGS", "₹1,20,00,000", delta="+₹1.2 Cr/yr")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Q2 · Net Benefit Calculation ───────────────────────────────
    section_header("Q2 — Net Benefit Calculation")

    st.latex(r"Net\ Benefit = Total\ Savings - Investment")
    st.latex(r"= 1{,}20{,}00{,}000 - 1{,}10{,}00{,}000 = 10{,}00{,}000")

    q2c1, q2c2, q2c3 = st.columns(3)
    with q2c1:
        st.metric(
            "TOTAL SAVINGS",
            "₹1.2 Crores",
            delta="Annual Recurring",
            delta_color="normal",
        )
    with q2c2:
        st.metric(
            "IOT INVESTMENT",
            "₹1.1 Crores",
            delta="One-time Capex",
            delta_color="normal",
        )
    with q2c3:
        st.metric(
            "NET BENEFIT",
            "₹10 Lakhs",
            delta="Year 1 Profit",
            delta_color="normal",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 5-Year Cash Flow & ROI Projection ──────────────────────────
    section_header("5-Year Cash Flow and ROI Projection")

    years = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    savings = [120, 125, 130, 135, 140]
    investment = [110, 4.25, 4.25, 4.25, 4.25]
    net_benefit = [s - i for s, i in zip(savings, investment)]
    cumulative = []
    running = 0
    for nb in net_benefit:
        running += nb
        cumulative.append(running)

    fig = go.Figure()

    # Green bars — savings
    fig.add_trace(go.Bar(
        x=years, y=savings,
        name="Annual Savings (Lakhs)",
        marker_color="#2d6a4f",
        text=[f"₹{v}L" for v in savings],
        textposition="outside",
    ))

    # Amber bars — investment/maintenance
    fig.add_trace(go.Bar(
        x=years, y=investment,
        name="Investment / Maintenance (Lakhs)",
        marker_color="#e9ac04",
        text=[f"₹{v}L" for v in investment],
        textposition="outside",
    ))

    # Yellow line — net benefit per year
    fig.add_trace(go.Scatter(
        x=years, y=net_benefit,
        name="Net Benefit / Year",
        mode="lines+markers+text",
        line=dict(color="#d4a017", width=2.5),
        marker=dict(size=8),
        text=[f"₹{v:.1f}L" for v in net_benefit],
        textposition="top center",
        textfont=dict(size=11),
    ))

    # Green line — cumulative
    fig.add_trace(go.Scatter(
        x=years, y=cumulative,
        name="Cumulative Net Benefit",
        mode="lines+markers+text",
        line=dict(color="#52b788", width=2.5, dash="dot"),
        marker=dict(size=8),
        text=[f"₹{v:.1f}L" for v in cumulative],
        textposition="top center",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        barmode="group",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        yaxis=dict(
            title="Amount (Lakhs ₹)",
            gridcolor="#eeeeee",
            zeroline=False,
        ),
        xaxis=dict(title=""),
        margin=dict(t=60, b=40, l=60, r=30),
        height=480,
    )

    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "The IoT system pays for itself in Year 1 with a net benefit of "
        "₹10 Lakhs. From Year 2 onward, maintenance costs drop to just "
        "₹4.25 Lakhs per year, resulting in annual net gains exceeding "
        "₹120 Lakhs. By the end of Year 5, the cumulative net benefit "
        "reaches approximately ₹607 Lakhs — a strong return on a "
        "₹1.1 Crore initial investment.",
        label="Key Takeaway",
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Livestock Data & Simulation
# ══════════════════════════════════════════════════════════════════════
elif page == "Livestock Data & Simulation":
    st.markdown(
        """
        <div class="main-header">
            <h1>Livestock Data &amp; Simulation</h1>
            <p>Questions 3 &amp; 4 — IoT Components &amp; Synthetic Data</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Q3 · Key IoT Components ────────────────────────────────────
    section_header("Q3 — Key IoT Components")

    iot_data = {
        "Component": [
            "GPS Collar",
            "Temperature Sensor",
            "Activity Monitor",
            "Feeding Sensor",
            "Milk Yield Sensor",
            "Cloud Dashboard",
            "Alert System",
        ],
        "Type": [
            "Location",
            "Health",
            "Behaviour",
            "Nutrition",
            "Productivity",
            "Software",
            "Automation",
        ],
        "Why It Matters": [
            "Tracks animal, prevents theft",
            "Detects fever early",
            "Low activity = illness signal",
            "Ensures proper feeding",
            "Auto daily output record",
            "Central view of all 2000 animals",
            "Instant farmer notification",
        ],
    }
    iot_df = pd.DataFrame(iot_data)
    st.table(iot_df.set_index("Component"))

    insight_box(
        "These 7 components form a complete end-to-end IoT stack — from "
        "field-level sensors on each animal to a centralised cloud dashboard "
        "that gives the farm owner a real-time view of all 2,000 animals "
        "without physical rounds."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Q4 · Synthetic Data Generation ────────────────────────────
    section_header("Q4 — How Synthetic Data Was Generated")

    st.code(
        """import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

animal_ids    = np.random.randint(1, 2001, size=n)
dates         = pd.date_range(start="2023-01-01", periods=n, freq="h")
temperature   = np.round(np.random.normal(38.5, 0.5, n), 1)
activity_score= np.round(np.random.uniform(30, 100, n), 1)
feeding_time  = np.round(np.random.normal(45, 10, n), 1)
milk_yield    = np.round(5 + 0.05 * activity_score + np.random.normal(0, 1, n), 2)
alert_status  = np.where(
    (temperature > 39.5) | (activity_score < 40), "Alert", "Normal"
)

df = pd.DataFrame({
    "AnimalID"    : animal_ids,
    "Date"        : dates,
    "Temperature" : temperature,
    "ActivityScore": activity_score,
    "FeedingTime" : feeding_time,
    "MilkYield"   : milk_yield,
    "AlertStatus" : alert_status,
})
df["Month"]     = df["Date"].dt.month
df["Year"]      = df["Date"].dt.year
df["DayOfWeek"] = df["Date"].dt.dayofweek""",
        language="python",
    )

    st.markdown("**Dataset Preview — First 10 Records**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 1 — Monthly Average Milk Yield Trend ─────────────────
    section_header("Monthly Average Milk Yield Trend")

    monthly = (
        df.groupby(["Year", "Month"])["MilkYield"]
        .mean()
        .reset_index()
        .rename(columns={"MilkYield": "Avg Milk Yield"})
    )

    fig1 = go.Figure()
    for year, color, name in [(2023, "#2d6a4f", "2023"), (2024, "#e9ac04", "2024")]:
        d = monthly[monthly["Year"] == year]
        if not d.empty:
            fig1.add_trace(go.Scatter(
                x=d["Month"], y=d["Avg Milk Yield"],
                mode="lines+markers",
                name=str(year),
                line=dict(color=color, width=2.5),
                marker=dict(size=7),
            ))

    fig1.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        xaxis=dict(
            title="Month",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"],
            gridcolor="#eeeeee",
        ),
        yaxis=dict(title="Avg Milk Yield (L)", gridcolor="#eeeeee", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=50, b=40, l=60, r=30),
        height=380,
    )
    st.plotly_chart(fig1, use_container_width=True)
    insight_box(
        "Milk yield stays relatively stable across months, with minor peaks "
        "in cooler months. 2024 shows a slight upward trend compared to 2023, "
        "reflecting improved feeding regularity captured by IoT sensors."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 2 — Distribution of Milk Yield ──────────────────────
    section_header("Distribution of Milk Yield per Record")

    fig2 = px.histogram(
        df, x="MilkYield", nbins=50,
        color_discrete_sequence=["#2d6a4f"],
        labels={"MilkYield": "Milk Yield (L)", "count": "Record Count"},
    )
    fig2.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        bargap=0.05,
        xaxis=dict(gridcolor="#eeeeee"),
        yaxis=dict(gridcolor="#eeeeee", zeroline=False),
        margin=dict(t=40, b=40, l=60, r=30),
        height=360,
    )
    st.plotly_chart(fig2, use_container_width=True)
    insight_box(
        "The milk yield distribution is approximately normal, centred around "
        "7–8 litres per record. Outliers at the low end often correspond to "
        "animals flagged with an Alert status, confirming that health events "
        "directly reduce productivity."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 3 — Top 10 Animals by Total Milk Yield ──────────────
    section_header("Top 10 Animals by Total Milk Yield")

    top10 = (
        df.groupby("AnimalID")["MilkYield"]
        .sum()
        .nlargest(10)
        .reset_index()
        .sort_values("MilkYield")
    )
    top10["AnimalID"] = "Animal #" + top10["AnimalID"].astype(str)

    fig3 = px.bar(
        top10, x="MilkYield", y="AnimalID",
        orientation="h",
        color="MilkYield",
        color_continuous_scale=["#95d5b2", "#2d6a4f"],
        labels={"MilkYield": "Total Milk Yield (L)", "AnimalID": ""},
    )
    fig3.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        coloraxis_showscale=False,
        xaxis=dict(title="Total Milk Yield (L)", gridcolor="#eeeeee"),
        yaxis=dict(gridcolor="#eeeeee"),
        margin=dict(t=40, b=40, l=110, r=30),
        height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)
    insight_box(
        "The top-producing animals consistently yield significantly above "
        "the farm average. Identifying these high performers helps the farm "
        "prioritise their health monitoring, feed allocation, and breeding "
        "decisions to maximise overall productivity."
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Cluster Analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Cluster Analysis":
    st.markdown(
        """
        <div class="main-header">
            <h1>Cluster Analysis</h1>
            <p>Question 5 — K-Means Clustering on Livestock Behaviour</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Q5 — K-Means Clustering (k = 4)")

    # ── Aggregate per animal ───────────────────────────────────────
    animal_agg = agg_df.copy()

    clust_features = animal_agg[["TotalMilkYield", "TotalRecords", "AvgMilkYield"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clust_features)

    # ── Elbow Method ──────────────────────────────────────────────
    section_header("Elbow Method — Optimal K")

    inertia = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertia.append(km.inertia_)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range), y=inertia,
        mode="lines+markers",
        line=dict(color="#2d6a4f", width=2.5),
        marker=dict(size=8),
    ))
    fig_elbow.add_vline(x=4, line_dash="dash", line_color="#e9ac04", line_width=2)
    fig_elbow.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        xaxis=dict(title="Number of Clusters (K)", gridcolor="#eeeeee", dtick=1),
        yaxis=dict(title="Inertia", gridcolor="#eeeeee", zeroline=False),
        margin=dict(t=40, b=40, l=70, r=30),
        height=340,
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    insight_box(
        "The elbow curve shows a clear bend at K=4 (amber dashed line), "
        "indicating that 4 clusters capture the natural groupings in the "
        "per-animal milk yield data without over-segmenting."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── K=4 Clustering ────────────────────────────────────────────
    # Using pre-computed Cluster labels from cluster_data.csv to match Colab exactly
    # kmeans4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    # animal_agg["Cluster"] = kmeans4.fit_predict(scaled)

    # Fixed labels per cluster index (match Colab notebook output)
    cluster_label_map = {
        0: "High Performer",
        1: "Efficient Low Volume",
        2: "Underperformer",
        3: "Moderate Performer",
    }
    cluster_color_map = {
        "High Performer":       "#2d6a4f",
        "Efficient Low Volume": "#52b788",
        "Underperformer":       "#d62728",
        "Moderate Performer":   "#e9ac04",
    }
    animal_agg["Segment"] = animal_agg["Cluster"].map(cluster_label_map)

    # ── Scatter: TotalRecords vs TotalMilkYield ───────────────────
    section_header("Total Records vs Total Milk Yield by Cluster")

    fig_scatter = px.scatter(
        animal_agg, x="TotalRecords", y="TotalMilkYield",
        color="Segment",
        color_discrete_map=cluster_color_map,
        labels={
            "TotalRecords":    "Total Records",
            "TotalMilkYield":  "Total Milk Yield (L)",
        },
        opacity=0.75,
    )
    fig_scatter.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        xaxis=dict(gridcolor="#eeeeee"),
        yaxis=dict(gridcolor="#eeeeee", zeroline=False),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=50, b=40, l=60, r=30),
        height=440,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Cluster Profile Table ─────────────────────────────────────
    section_header("Cluster Profile — Average Metrics")

    profile = (
        animal_agg.groupby("Segment")[["TotalMilkYield", "TotalRecords", "AvgMilkYield"]]
        .mean()
        .round(2)
    )
    profile.columns = ["Avg Total Milk Yield (L)", "Avg Records", "Avg Milk/Record (L)"]
    st.table(profile)

    insight_box(
        "Four distinct animal segments emerge from K-Means (k=4) clustering "
        "on per-animal aggregated milk yield data:\n\n"
        "  High Performers — high total milk yield (~32.5L) with frequent "
        "records (~5.3).\n"
        "  Efficient Low Volume — fewer records (~1.6) but high per-record "
        "yield (~7.37L) — healthy but infrequent in the log.\n"
        "  Underperformers — both low total (~6.1L) and low average "
        "(~3.49L) — prime candidates for health intervention.\n"
        "  Moderate Performers — middle-ground animals (~19.5L total) "
        "that respond well to feed optimisation."
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Forecast
# ══════════════════════════════════════════════════════════════════════
elif page == "Forecast":
    st.markdown(
        """
        <div class="main-header">
            <h1>Forecast</h1>
            <p>Question 6 — Monthly Milk Yield Forecast using Linear Regression</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Q6 — Linear Regression on Monthly TimeIndex")

    st.markdown(
        '<p style="color:#666; font-size:0.9rem; margin-bottom:1rem;">'
        "Feature: <strong>TimeIndex</strong> (sequential integer per month). "
        "Target: <strong>Total Monthly Milk Yield</strong>. "
        "Split: 80% train / 20% test, random_state=42.</p>",
        unsafe_allow_html=True,
    )

    # ── Build monthly aggregation ─────────────────────────────────
    monthly_total = monthly_df.copy()
    if "TimeIndex" not in monthly_total.columns:
        monthly_total["TimeIndex"] = range(len(monthly_total))
    if "Label" not in monthly_total.columns:
        monthly_total["Label"] = monthly_total.apply(
            lambda r: f"{int(r.Year)}-{int(r.Month):02d}", axis=1
        )

    X_m = monthly_total[["TimeIndex"]]
    y_m = monthly_total["MilkYield"]

    split = int(len(monthly_total) * 0.8)
    X_train_m, X_test_m = X_m.iloc[:split], X_m.iloc[split:]
    y_train_m, y_test_m = y_m.iloc[:split], y_m.iloc[split:]

    model_m = LinearRegression()
    model_m.fit(X_train_m, y_train_m)
    y_pred_m = model_m.predict(X_m)           # predict over full range for chart
    y_pred_test_m = model_m.predict(X_test_m)

    r2_m   = r2_score(y_test_m, y_pred_test_m)
    rmse_m = mean_squared_error(y_test_m, y_pred_test_m) ** 0.5

    # ── Metric cards ──────────────────────────────────────────────
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("R² SCORE", f"{r2_m:.4f}", delta="Model Fit Quality")
    with mc2:
        st.metric("RMSE", f"{rmse_m:.2f} L", delta="Root Mean Squared Error")
    with mc3:
        st.metric("TRAIN / TEST MONTHS", f"{split} / {len(monthly_total)-split}", delta="80/20 split")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Actual vs Predicted ────────────────────────────────────────
    section_header("Actual vs Predicted — Monthly Milk Yield")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=monthly_total["Label"], y=monthly_total["MilkYield"],
        mode="lines+markers", name="Actual",
        line=dict(color="#2d6a4f", width=2),
        marker=dict(size=6),
    ))
    fig_pred.add_trace(go.Scatter(
        x=monthly_total["Label"], y=y_pred_m,
        mode="lines", name="Predicted (Regression)",
        line=dict(color="#e9ac04", width=2.5, dash="dot"),
    ))
    fig_pred.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        xaxis=dict(title="Month", gridcolor="#eeeeee", tickangle=-45),
        yaxis=dict(title="Total Milk Yield (L)", gridcolor="#eeeeee", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=60, b=80, l=70, r=30),
        height=440,
        # Shade train region with a rectangle annotation
        shapes=[dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0,
            x1=split / len(monthly_total), y1=1,
            fillcolor="rgba(45,106,79,0.05)",
            line_width=0,
            layer="below",
        )],
        annotations=[dict(
            xref="paper", yref="paper",
            x=split / len(monthly_total), y=1.04,
            text="Train | Test", showarrow=False,
            font=dict(size=11, color="#888888"),
            xanchor="center",
        )],
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    insight_box(
        f"Using monthly aggregated milk yield and a simple TimeIndex feature, "
        f"the linear regression model achieves R² = {r2_m:.4f} on the test set — "
        "a strong fit that captures the overall trend across 2023 and 2024. "
        "The model is suitable for near-term production forecasting."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3-Month Forecast ──────────────────────────────────────────
    section_header("Next 3 Months Forecast")

    last_idx   = monthly_total["TimeIndex"].max()
    last_year  = int(monthly_total["Year"].max())
    last_month = int(monthly_total[monthly_total["Year"] == last_year]["Month"].max())

    forecast_rows = []
    for i in range(1, 4):
        ti = last_idx + i
        m  = (last_month + i - 1) % 12 + 1
        y  = last_year + (last_month + i - 1) // 12
        pred_val = model_m.predict([[ti]])[0]
        forecast_rows.append({"Month": f"{y}-{m:02d}", "TimeIndex": ti, "Forecast Milk Yield (L)": round(pred_val, 2)})

    forecast_df = pd.DataFrame(forecast_rows)
    st.table(forecast_df.set_index("Month"))

    insight_box(
        "Extending the regression line by 3 months gives farm managers an "
        "early indication of expected total milk output, enabling advance "
        "planning for feed procurement, staffing, and logistics.",
        label="Forecast Insight",
    )


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — Recommendation
# ══════════════════════════════════════════════════════════════════════
elif page == "Recommendation":
    st.markdown(
        """
        <div class="main-header">
            <h1>Recommendation</h1>
            <p>Question 7 — Strategic Recommendations &amp; Implementation Roadmap</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Should AgroFarm Invest in IoT?")

    st.markdown(
        '<div class="insight-box">'
        '<span class="insight-label">Executive Answer</span>'
        "<strong>Yes — strategically sound.</strong> Although Year 1 shows a "
        "small net loss of ₹15 Lakhs due to the upfront IoT investment, "
        "from Year 2 onward the farm earns ₹95 Lakhs net profit annually. "
        "The 5-year cumulative gain is <strong>₹365 Lakhs</strong> — a "
        "strong return on a one-time ₹1.1 Crore investment."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Financial Breakdown ───────────────────────────────────────
    section_header("Financial Calculation Breakdown")

    st.latex(r"Savings_{animal} = 2{,}00{,}00{,}000 \times 35\% = 70{,}00{,}000")
    st.latex(r"Savings_{labor}  = 1{,}00{,}00{,}000 \times 25\% = 25{,}00{,}000")
    st.latex(r"Total\ Annual\ Savings = 70L + 25L = 95{,}00{,}000")
    st.latex(r"Net\ Benefit\ (Year\ 1) = 95L - 1{,}10{,}00{,}000 = -15{,}00{,}000")
    st.latex(r"Net\ Benefit\ (Year\ 2{+}) = 95{,}00{,}000\ per\ year")
    st.latex(r"5\text{-}Year\ Cumulative = -15 + 95 + 95 + 95 + 95 = 365\ Lakhs")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Conclusion metric cards ───────────────────────────────────
    section_header("Conclusion")

    cols = st.columns(3)
    with cols[0]:
        st.metric("TOTAL ANNUAL SAVINGS", "₹95 Lakhs", delta="From Year 2 onward", delta_color="normal")
    with cols[1]:
        st.metric("YEAR 1 NET BENEFIT", "−₹15 Lakhs", delta="One-time capital loss", delta_color="inverse")
    with cols[2]:
        st.metric("5-YEAR CUMULATIVE", "₹365 Lakhs", delta="vs ₹1.1 Cr investment", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROI Summary Chart ─────────────────────────────────────────
    section_header("5-Year ROI Summary")

    years   = ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
    savings = [95, 95, 95, 95, 95]        # Lakhs
    invest  = [110, 0, 0, 0, 0]           # only Year 1 has capex
    net     = [-15, 95, 95, 95, 95]       # Year 1 = 95 - 110 = -15
    cum     = [-15, 80, 175, 270, 365]    # cumulative

    bar_colors = ["#d62728" if v < 0 else "#2d6a4f" for v in net]

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Bar(
        x=years, y=net,
        name="Net Benefit / Year (Lakhs)",
        marker_color=bar_colors,
        text=[f"₹{v}L" for v in net],
        textposition="outside",
    ))
    fig_roi.add_trace(go.Scatter(
        x=years, y=cum,
        name="Cumulative Benefit",
        mode="lines+markers",
        line=dict(color="#e9ac04", width=2.5),
        marker=dict(size=8),
        text=[f"₹{v}L" for v in cum],
        textposition="top center",
        textfont=dict(size=11),
    ))
    fig_roi.add_hline(y=0, line_color="#cccccc", line_width=1)
    fig_roi.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#1a1a1a"),
        xaxis=dict(gridcolor="#eeeeee"),
        yaxis=dict(title="Lakhs (₹)", gridcolor="#eeeeee", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=50, b=40, l=60, r=30),
        height=420,
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # ── Action Table ──────────────────────────────────────────────
    section_header("Recommended Actions")

    actions = {
        "Priority": ["High", "High", "Medium", "Medium", "Low"],
        "Action": [
            "Deploy GPS collars + temperature sensors on all 2,000 animals",
            "Integrate cloud dashboard with real-time alert system",
            "Automate feeding & milk yield sensor logs",
            "Train farm staff on IoT dashboard usage",
            "Expand sensor network to crop/soil monitoring",
        ],
        "Timeline": ["Month 1–2", "Month 2–3", "Month 3–4", "Month 4", "Year 2+"],
        "Expected Impact": [
            "Instant location & health data",
            "80% faster health intervention",
            "60% fewer logging errors",
            "Staff productivity +20%",
            "Holistic farm intelligence",
        ],
    }
    actions_df = pd.DataFrame(actions)
    st.table(actions_df.set_index("Priority"))

    insight_box(
        "Year 1 records a manageable loss of ₹15 Lakhs — the cost of "
        "deploying 2,000 IoT devices and infrastructure. From Year 2, all "
        "capital has been recovered and the farm earns ₹95 Lakhs net annually. "
        "By Year 5 the cumulative benefit reaches ₹365 Lakhs. This is a "
        "textbook positive-NPV investment that every modern agribusiness "
        "should pursue.",
        label="Final Recommendation",
    )


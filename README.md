# AgroFarm IoT Dashboard

This repository contains the interactive dashboard for **AgroFarm Analytics**, a data-driven approach to livestock tracking and management using IoT sensors. This dashboard was developed as part of Case Study 39 (Business Studies · Semester IV) to evaluate the financial feasibility, behavioral patterns, and productivity impact of migrating from a manual monitoring system to an automated IoT stack.

## Live Demo
https://agrofarm-dashboard-qjfzljazld54fnh4hg8jfu.streamlit.app/

## Overview
The AgroFarm IoT Dashboard leverages stream processing and machine learning models to identify actionable insights from a cattle herd of 2,000 animals. The platform addresses several key aspects:

1. **Financial Analysis:** ROI projection comparing the ₹2 Cr manual livestock monitoring loss against a ₹1.1 Cr IoT investment, achieving a positive cash flow in Year 1.
2. **Livestock Sensors:** End-to-end integration of GPS collars, temperature monitoring, and automated milking metrics.
3. **Cluster Analysis (K-Means):** Segmentation of cows into four distinct productivity tiers (e.g., "High Performers," "Underperformers") to target early health interventions.
4. **Production Forecast (Linear Regression):** Forecasting exact monthly milk yields to aid in supply chain and logistics planning.

## Tech Stack
- **Dashboard Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (K-Means clustering, Linear Regression)
- **Visualizations:** `plotly`, `matplotlib`, `seaborn`

## Local Setup

**Clone the repository:**
```bash
git clone https://github.com/vivek-419/agrofarm-dashboard.git
cd agrofarm-dashboard
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the app:**
```bash
streamlit run app.py
```

## Dashboard Contents

- `app.py`: The main Streamlit application containing the multi-page layout and integrated visualizations.
- `requirements.txt`: Python package dependencies.
- `livestock_data.csv`: Source dataset for exploratory data analysis (EDA).
- `cluster_data.csv`: Pre-aggregated data used for the K-Means behavior clustering.
- `monthly_yield.csv`: Monthly time-series data used for the yield forecasting model.

## Case Study Conclusion
The case study definitively recommends the IoT deployment. While Year 1 experiences a minor capital hurdle, Year 2 onwards generates ₹95 Lakhs in annual net profit, recovering the entire infrastructure cost and unlocking a cumulative 5-year benefit of ₹365 Lakhs.

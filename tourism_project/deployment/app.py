'''Streamlit Community Cloud entry point: customer inputs become a scoring DataFrame.'''

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Support Streamlit's nested entry point as well as invocation from the repository root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tourism_project.deployment.inference import load_bundle, predict
from tourism_project.model_building.schema import CHOICES, DEPLOYMENT

st.set_page_config(page_title="Visit with Us | Campaign Planning", page_icon=":material/travel_explore:", layout="wide")


@st.cache_resource
def cached_bundle(version):
    return load_bundle()


st.title("Visit with Us")
st.caption("Wellness tourism | Campaign planning")
try:
    version = (DEPLOYMENT / "model_metadata.json").stat().st_mtime_ns
    model, metadata = cached_bundle(version)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Model unavailable: {exc}")
    st.stop()

single_tab, batch_tab, performance_tab = st.tabs(["Customer", "Campaign list", "Model performance"])
with single_tab:
    with st.form("customer_form"):
        left, middle, right = st.columns(3)
        with left:
            st.subheader("Customer profile")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Monthly income", min_value=0, max_value=1000000, value=24000, step=500)
            occupation = st.selectbox("Occupation", CHOICES["Occupation"])
            designation = st.selectbox("Designation", CHOICES["Designation"])
            gender = st.selectbox("Gender", CHOICES["Gender"])
        with middle:
            st.subheader("Travel plans")
            persons = st.number_input("Total travellers", min_value=1, max_value=20, value=3)
            children = st.number_input("Children below age 5", min_value=0, max_value=20, value=1)
            trips = st.number_input("Trips per year", min_value=0, max_value=50, value=3)
            stars = st.selectbox("Preferred hotel stars", CHOICES["PreferredPropertyStar"])
        with right:
            st.subheader("Additional details")
            city = st.selectbox("City tier", CHOICES["CityTier"])
            marital = st.selectbox("Marital status", CHOICES["MaritalStatus"])
            passport = st.checkbox("Valid passport", value=True)
            car = st.checkbox("Owns a car", value=False)
        submitted = st.form_submit_button("Assess customer", type="primary", icon=":material/query_stats:")
    if submitted:
        inputs = pd.DataFrame([{"Age": age, "MonthlyIncome": income, "Occupation": occupation,
            "Designation": designation, "Gender": gender, "NumberOfPersonVisiting": persons,
            "NumberOfChildrenVisiting": children, "NumberOfTrips": trips, "PreferredPropertyStar": stars,
            "CityTier": city, "MaritalStatus": marital, "Passport": int(passport), "OwnCar": int(car)}])
        try:
            result = predict(inputs, model, metadata).iloc[0]
            score_column, priority_column = st.columns(2)
            score_column.metric("Purchase score", f"{result.purchase_score:.1%}")
            priority_column.metric("Campaign priority", "Priority contact" if result.priority_contact else "Standard nurture")
            st.caption(f"Contact threshold: {metadata['threshold']:.2f}. Scores rank interest; they are not calibrated purchase probabilities.")
        except ValueError as exc:
            st.error(str(exc))

with batch_tab:
    uploaded = st.file_uploader("Customer list", type=["csv"])
    sample = ROOT / "tourism_project" / "data" / "sample_customers.csv"
    if sample.exists():
        st.download_button("Download input template", sample.read_bytes(), "sample_customers.csv", "text/csv", icon=":material/download:")
    if uploaded is not None:
        try:
            customer_frame = pd.read_csv(uploaded)
            scored = predict(customer_frame, model, metadata)
            result_frame = pd.concat([customer_frame, scored], axis=1).sort_values("purchase_score", ascending=False)
            st.dataframe(result_frame, hide_index=True, use_container_width=True)
            st.download_button("Download ranked customers", result_frame.to_csv(index=False), "ranked_customers.csv", "text/csv", icon=":material/download:")
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))

with performance_tab:
    cols = st.columns(3)
    cols[0].metric("Test average precision", f"{metadata['test_metrics']['average_precision']:.3f}")
    cols[1].metric("Buyer recall", f"{metadata['test_metrics']['recall']:.1%}")
    cols[2].metric("Top 20% lift", f"{metadata['top_20pct_lift']:.2f}x")
    plot = ROOT / "tourism_project" / "reports" / "test_performance.png"
    if plot.exists():
        st.image(str(plot), caption=f"Held-out evaluation: {metadata['test_rows']} customers")

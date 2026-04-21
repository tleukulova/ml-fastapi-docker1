"""
app.py - Streamlit frontend for the Titanic Survival Prediction API.
Sends passenger data to the FastAPI service running on port 1912
and displays the prediction result in a user-friendly interface.
"""

import streamlit as st
import requests

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

API_URL = "http://localhost:1912"

# ──────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

st.title("🚢 Titanic Survival Predictor")
st.markdown(
    "Fill in the passenger details below and click **Predict** "
    "to find out whether this person would have survived."
)
st.divider()

# ──────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────

with st.form("passenger_form"):
    st.subheader("Passenger Information")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            index=2,
            help="1 = First class, 2 = Second class, 3 = Third class",
        )

        sex_label = st.radio(
            "Sex",
            options=["Female", "Male"],
            index=1,
            horizontal=True,
        )
        sex = 0 if sex_label == "Female" else 1

        age = st.slider(
            "Age",
            min_value=0.5,
            max_value=80.0,
            value=22.0,
            step=0.5,
        )

        fare = st.number_input(
            "Fare (£)",
            min_value=0.0,
            max_value=600.0,
            value=7.25,
            step=0.5,
            help="Ticket price in British pounds",
        )

    with col2:
        sibsp = st.number_input(
            "Siblings / Spouses aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

        parch = st.number_input(
            "Parents / Children aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )

        embarked_label = st.selectbox(
            "Port of Embarkation",
            options=["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
            index=2,
        )
        embarked_map = {
            "Cherbourg (C)": 0,
            "Queenstown (Q)": 1,
            "Southampton (S)": 2,
        }
        embarked = embarked_map[embarked_label]

    submitted = st.form_submit_button("🔍 Predict Survival", use_container_width=True)

# ──────────────────────────────────────────────
# PREDICTION REQUEST
# ──────────────────────────────────────────────

if submitted:
    payload = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": fare,
        "Embarked": embarked,
    }

    try:
        with st.spinner("Sending request to the API..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            response.raise_for_status()

        result = response.json()
        survived = result["survived"]
        label = result["survived_label"]
        prob = result["probability"]

        st.divider()
        st.subheader("Prediction Result")

        if survived == 1:
            st.success(f"✅ **{label}**")
        else:
            st.error(f"❌ **{label}**")

        st.metric(
            label="Survival Probability",
            value=f"{prob * 100:.1f}%",
        )

        # Visual probability bar
        st.progress(prob, text=f"Confidence: {prob * 100:.1f}%")

        # Show submitted data in expander
        with st.expander("📋 Submitted passenger data"):
            st.json(payload)

    except requests.exceptions.ConnectionError:
        st.error(
            "❗ Could not connect to the API. "
            "Make sure the FastAPI server is running on port 1912."
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"❗ API returned an error: {e.response.text}")
    except Exception as e:
        st.error(f"❗ Unexpected error: {e}")

# ──────────────────────────────────────────────
# SIDEBAR INFO
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This app uses a **Random Forest** classifier trained on the
        [Titanic dataset](https://www.kaggle.com/c/titanic) to predict
        whether a passenger would have survived.

        **API Status**
        """
    )
    try:
        health = requests.get(f"{API_URL}/", timeout=2)
        if health.status_code == 200:
            st.success("🟢 API is online")
        else:
            st.warning("🟡 API returned unexpected status")
    except Exception:
        st.error("🔴 API is offline")

    st.divider()
    st.caption("FastAPI  •  MLflow  •  Streamlit  •  Docker")

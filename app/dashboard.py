import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import pandas as pd
import pickle

from streamlit_js_eval import get_geolocation
from src.weather import get_weather


# ======================
# Page Config
# ======================

st.set_page_config(
    page_title="SolarSense AI",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ SolarSense AI")

st.markdown(
    "### Intelligent Solar Power Prediction System"
)


# ======================
# Load Models
# ======================

@st.cache_resource
def load_model(name):

    with open(
        f"models/{name}.pkl",
        "rb"
    ) as f:

        return pickle.load(f)


@st.cache_resource
def load_anomaly():

    with open(
        "models/anomaly.pkl",
        "rb"
    ) as f:

        return pickle.load(f)


anomaly_model = load_anomaly()


# ======================
# Sidebar
# ======================

st.sidebar.header(
    "Prediction Settings"
)

use_location = st.sidebar.toggle(
    "Use Current Location"
)

weather_data = None


# ======================
# Location
# ======================

if use_location:

    location = get_geolocation()

    if location:

        lat = location["coords"]["latitude"]

        lon = location["coords"]["longitude"]

        weather_data = get_weather(
            lat,
            lon
        )

        st.success(
            "📍 Current location detected"
        )

        col1,col2=st.columns(2)

        with col1:

            st.metric(
                "Temperature",
                f"{weather_data['temp']}°C"
            )

        with col2:

            st.metric(
                "Humidity",
                f"{weather_data['humidity']}%"
            )



# ======================
# Input Section
# ======================

st.sidebar.header(
    "Input Parameters"
)

hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    10
)

target = st.sidebar.slider(
    "Target Temperature",
    0.0,
    60.0,
    40.0
)

voltage = st.sidebar.slider(
    "Voltage",
    0.0,
    30.0,
    18.0
)

current = st.sidebar.slider(
    "Current",
    0.0,
    5.0,
    0.2
)


# Manual weather only if GPS OFF

if not use_location:

    ambient = st.sidebar.slider(
        "Ambient Temperature",
        0.0,
        50.0,
        30.0
    )

    humidity = st.sidebar.slider(
        "Humidity",
        0,
        100,
        50
    )

else:

    ambient = weather_data["temp"]

    humidity = weather_data["humidity"]


season = st.sidebar.selectbox(
    "Season",
    [
        "rainy",
        "summer",
        "winter"
    ]
)

time_period = st.sidebar.selectbox(
    "Time Period",
    [
        "Morning",
        "Afternoon",
        "Evening"
    ]
)

selected_model = st.sidebar.selectbox(
    "Prediction Model",
    [
        "LinearRegression",
        "RandomForest",
        "XGBoost"
    ]
)


# ======================
# Encoding
# ======================

season_map={

    "rainy":0,
    "summer":1,
    "winter":2
}

period_map={

    "Morning":0,
    "Afternoon":1,
    "Evening":2
}


# ======================
# Prediction Button
# ======================

if st.button(
    "⚡ Predict Power"
):

    temp_diff = (
        target
        -
        ambient
    )


    features=[[
        hour,
        ambient,
        target,
        humidity,
        season_map[season],
        temp_diff,
        period_map[time_period]
    ]]


    model=load_model(
        selected_model
    )


    predicted_power = model.predict(
        features
    )[0]


    # ======================
    # Prediction Section
    # ======================

    st.subheader(
        "⚡ Prediction Results"
    )

    col1,col2=st.columns(2)

    with col1:

        st.metric(
            "Predicted Power",
            f"{predicted_power:.2f} W"
        )



    with col2:

        if predicted_power > 40:

            status="Excellent ⚡"

        elif predicted_power > 20:

            status="Moderate"

        else:

            status="Poor"


        st.metric(
            "Performance",
            status
        )


    # ======================
    # Anomaly Detection
    # ======================

    st.subheader(
        "🚨 Panel Health"
    )


    anomaly_features=[[
        voltage,
        current,
        predicted_power,
        ambient,
        humidity
    ]]


    result=anomaly_model.predict(
        anomaly_features
    )


    if result[0]==-1:

        st.error(
            "⚠ Possible abnormal behavior detected"
        )

    else:

        st.success(
            "✅ Normal panel operation"
        )


# ======================
# Model Comparison
# ======================

st.subheader(
    "📊 Model Comparison"
)

results = pd.read_csv(
    "models/model_results.csv"
)

# Fix unnamed first column

if "Model" not in results.columns:

    results.rename(
        columns={
            results.columns[0]:"Model"
        },
        inplace=True
    )

chart_data = results[
    [
        "Model",
        "R2"
    ]
]

st.bar_chart(
    chart_data.set_index(
        "Model"
    )
)

st.dataframe(
    results
)
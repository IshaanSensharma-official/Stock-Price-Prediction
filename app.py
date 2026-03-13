import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("Stock Price Prediction using LSTM")

# -------------------------------
# USER INPUT
# -------------------------------
stock = st.text_input(
    "Enter Stock Ticker (Example: AAPL, TSLA, RELIANCE.NS)",
    "AAPL"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model("model.keras")

# -------------------------------
# DOWNLOAD DATA
# -------------------------------
if st.button("Predict"):

    data = yf.download(
        tickers=stock,
        period="10y",
        interval="1d",
        auto_adjust=False
    )

    if data.empty:
        st.error("Invalid ticker symbol")
        st.stop()

    data.reset_index(inplace=True)

    st.subheader("Raw Data")
    st.dataframe(data.tail())

    # -------------------------------
    # PRICE CHART
    # -------------------------------

    st.subheader("Closing Price")

    fig = plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Close'])
    plt.xlabel("Date")
    plt.ylabel("Price")
    st.pyplot(fig)

    # -------------------------------
    # MOVING AVERAGES
    # -------------------------------

    ma100 = data['Close'].rolling(100).mean()
    ma200 = data['Close'].rolling(200).mean()

    st.subheader("Moving Averages")

    fig2 = plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label="Close")
    plt.plot(ma100, label="100 MA")
    plt.plot(ma200, label="200 MA")
    plt.legend()
    st.pyplot(fig2)

    # -------------------------------
    # TRAIN TEST SPLIT
    # -------------------------------

    data.dropna(inplace=True)

    data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.8)])
    data_test = pd.DataFrame(data['Close'][int(len(data)*0.8):])

    scaler = MinMaxScaler(feature_range=(0,1))
    data_train_scale = scaler.fit_transform(data_train)

    # -------------------------------
    # TEST DATA PREPARATION
    # -------------------------------

    past_100_days = data_train.tail(100)

    final_df = pd.concat([past_100_days, data_test], ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # -------------------------------
    # PREDICTION
    # -------------------------------

    y_predicted = model.predict(x_test)

    scale_factor = 1 / scaler.scale_[0]

    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # -------------------------------
    # PLOT RESULTS
    # -------------------------------

    st.subheader("Prediction vs Original")

    fig3 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label="Original Price")
    plt.plot(y_predicted, 'r', label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    st.pyplot(fig3)

    st.success("Prediction completed")
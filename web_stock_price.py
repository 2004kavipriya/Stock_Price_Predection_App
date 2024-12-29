import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.title("📊 Stock Price Predictor App")

# Initialize session state for stock data
if "google_data" not in st.session_state:
    st.session_state.google_data = None

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Stock Selection", "Data Visualization", "Prediction"])

# Stock Selection Section
if section == "Stock Selection":
    st.header("📈 Stock Selection")
    stock = st.text_input("Enter the Stock ID", "GOOG")
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)

    if st.button("Load Data"):
        with st.spinner("Downloading stock data..."):
            st.session_state.google_data = yf.download(stock, start=start, end=end)
        if not st.session_state.google_data.empty:
            st.success("Stock data loaded successfully!")
        else:
            st.error("Failed to load stock data. Please check the stock symbol.")

    if st.session_state.google_data is not None:
        st.subheader("Stock Data")
        st.write(st.session_state.google_data)

# Data Visualization Section
elif section == "Data Visualization":
    if st.session_state.google_data is None or st.session_state.google_data.empty:
        st.warning("Please load stock data in the 'Stock Selection' section first!")
    else:
        st.header("📈 Data Visualization")
        st.markdown("Explore the stock price trends and moving averages.")
        google_data = st.session_state.google_data.copy()  # Create a copy to avoid modifying original data

        # Calculate moving averages
        google_data['MA100'] = google_data['Close'].rolling(window=100, min_periods=1).mean()
        google_data['MA200'] = google_data['Close'].rolling(window=200, min_periods=1).mean()
        google_data['MA250'] = google_data['Close'].rolling(window=250, min_periods=1).mean()

        # Visualization options
        st.subheader("Moving Averages")
        selected_ma = st.multiselect(
            "Select Moving Averages to Display",
            ["100 days", "200 days", "250 days"],
            default=["100 days", "200 days"],
        )

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot closing price
        ax.plot(google_data.index, google_data['Close'], 
                label='Close Price', 
                color='blue', 
                linewidth=2)
        
        # Plot selected moving averages
        ma_colors = {'100 days': 'orange', '200 days': 'green', '250 days': 'red'}
        ma_columns = {'100 days': 'MA100', '200 days': 'MA200', '250 days': 'MA250'}
        
        for ma in selected_ma:
            ax.plot(google_data.index, 
                   google_data[ma_columns[ma]], 
                   label=f'{ma} MA',
                   color=ma_colors[ma],
                   linewidth=1.5)

        # Customize the plot
        ax.set_title("Stock Prices and Moving Averages", fontsize=12, pad=20)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price", fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Display the plot
        st.pyplot(fig)

        # Display moving averages data
        if st.checkbox("Show Moving Averages Data"):
            ma_data = google_data[['Close'] + list(ma_columns.values())].tail(10)
            st.dataframe(ma_data.style.format("{:.2f}"))

# Prediction Section
elif section == "Prediction":
    if st.session_state.google_data is None or st.session_state.google_data.empty:
        st.warning("Please load stock data in the 'Stock Selection' section first!")
    else:
        st.header("📈 Stock Price Prediction")
        google_data = st.session_state.google_data

        # Load the pre-trained model
        model = load_model("Latest_stock_price_model.keras")

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(google_data['Close'].values.reshape(-1, 1))

        # Prepare the data for prediction
        x_data = []
        y_data = []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Make predictions
        predictions = model.predict(x_data)

        # Inverse scaling
        predictions = scaler.inverse_transform(predictions)
        y_data = scaler.inverse_transform(y_data)

        # Create DataFrame with aligned indices
        start_idx = 100  # Because we used 100 days for prediction
        plot_index = google_data.index[start_idx:]
        plotting_data = pd.DataFrame(
            {
                'original_test_data': y_data.reshape(-1),
                'predictions': predictions.reshape(-1)
            },
            index=plot_index
        )

        # Display the dataframe with original vs predicted data
        st.subheader("Original values vs Predicted values")
        st.write(plotting_data)

        # Plotting the original vs predicted close prices
        st.subheader("Original Close Price vs Predicted Close Price")
        fig = plt.figure(figsize=(15, 6))
        plt.plot(google_data.index[:start_idx], google_data['Close'][:start_idx], 'b', label='Training Data')
        plt.plot(plotting_data.index, plotting_data['original_test_data'], 'g', label='Original Test Data')
        plt.plot(plotting_data.index, plotting_data['predictions'], 'r', label='Predicted Values')
        plt.legend()
        st.pyplot(fig)
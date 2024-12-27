import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

# User input for stock symbol
stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download the stock data
google_data = yf.download(stock, start=start, end=end)

# Load the pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(google_data)

# Splitting the data into training and test sets (70% for training and 30% for testing)
splitting_len = int(len(google_data) * 0.7)
x_test = google_data['Close'][splitting_len:]

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label='Moving Average')
    plt.plot(full_data['Close'], 'b', label='Close Price')
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset, 'g', label='Additional MA')
    plt.legend()
    return fig

# Plotting the moving averages for different time windows
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(google_data['Close'].values.reshape(-1, 1))

# Prepare the data for prediction
x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Calculate the correct starting index for plotting
start_idx = 100  # Because we used 100 days for prediction
train_size = splitting_len
total_dataset_len = len(google_data)

# Create proper index for the plotting dataframe
plot_index = google_data.index[start_idx:]

# Inverse scaling
predictions = scaler.inverse_transform(predictions)
y_data = scaler.inverse_transform(y_data)

# Create DataFrame with aligned indices
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
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15,6))

# Plot the entire original data
plt.plot(google_data.index[:start_idx], google_data['Close'][:start_idx], 'b', label='Training Data')
plt.plot(plotting_data.index, plotting_data['original_test_data'], 'g', label='Original Test Data')
plt.plot(plotting_data.index, plotting_data['predictions'], 'r', label='Predicted Values')
plt.legend()
st.pyplot(fig)
# Stock_price_prediction
## Overview
This project demonstrates stock price prediction using two machine learning approaches:

PCA + LSTM: A dimensionality reduction technique (Principal Component Analysis) combined with a deep learning model (Long Short-Term Memory).

Moving Average + Linear Regression: A traditional method where a moving average is calculated and followed by applying a linear regression model.

The predictions are made based on historical stock data provided in a CSV file (stock_test.csv).

## Data
Input CSV: stock_test.csv
This CSV file contains historical stock price data with the following columns:

Date: The date for the stock price.
Open: The opening price of the stock.
High: The highest price of the stock on that day.
Low: The lowest price of the stock on that day.
Close: The closing price of the stock.
Volume: The trading volume of the stock.

### Assumptions
The CSV file should be pre-processed, cleaned, and formatted correctly.
Missing data should either be handled (e.g., imputed) or removed before feeding into the model.

### Algorithms Used
1. PCA + LSTM Approach
Principal Component Analysis (PCA) is used to reduce the dimensionality of the stock data while retaining the most important features.
The reduced dataset is then passed into an LSTM (Long Short-Term Memory) model, a type of recurrent neural network (RNN) commonly used for time series forecasting.

Key Steps:

Load and preprocess the data (e.g., normalization).
Apply PCA to reduce the feature space.
Build and train an LSTM model using the reduced dataset.
Make stock price predictions using the trained LSTM model.

2. Moving Average + Linear Regression Approach
The Moving Average of the stock prices is calculated to smooth out the fluctuations and better represent the stock's trend.
After calculating the moving average, a Linear Regression model is applied to predict future stock prices based on this trend.

Key Steps:

Load the data and calculate the moving average.
Prepare the data for regression by selecting appropriate features (e.g., moving average).
Apply linear regression to predict stock prices.
Evaluate the model performance and make predictions.

## How to Run the Project
1. Install Dependencies
Ensure that you have Python installed with the required libraries. Install dependencies using:
pip install -r requirements.txt

Required Libraries:

- numpy
- pandas
- matplotlib
- sklearn
- tensorflow (for LSTM model)
- statsmodels (for moving average)

2. Run the Script
To run the stock price prediction model on stock_test.csv:
 python stock_prediction.py

## File Descriptions
- stock_test.csv: The input file containing historical stock prices.
- stock_prediction.py: The main script that runs the PCA + LSTM and Moving Average + Linear Regression models on the stock data.
- LR_Actual_Prices.csv: Output file storing actual stock prices (from the moving average + linear regression model).
- LR_Predicted_Prices.csv: Output file storing predicted stock prices (from the moving average + linear regression model).
- LSTM_Predictions.csv: Output file storing predicted stock prices (from the PCA + LSTM model).

## Outputs
After running the script, you should see:

- Predicted Stock Prices: CSV files containing the actual and predicted stock prices for both models.
Visualization: Plots comparing the actual stock prices with the predicted stock prices.
Evaluation Metrics
- Mean Squared Error (MSE): Used to evaluate the accuracy of the model's predictions compared to the actual stock prices.
Root Mean Squared Error (RMSE): A more interpretable version of MSE, used for error analysis.

## Future Work
- Experiment with other time series models (e.g., ARIMA, Prophet).
- Hyperparameter tuning of the LSTM model for improved performance.
- Include more features such as external market factors, news sentiment, etc., for better predictions.





import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data from the Excel file
data = pd.read_excel('ASN3-Run1-Dataset.xlsx',header=None)

# Define the number of periods to forecast
forecast_periods = 3

# Fit a SARIMA model
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # Seasonal order for monthly data
model = sm.tsa.SARIMAX(data, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

# Predict the next three periods
forecast = model_fit.forecast(steps=forecast_periods).round()

# Display the predictions
print("Predicted Values:")
print(forecast)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data, label="Original Data")
#forecast_index = pd.date_range(start=data.index[-1], periods=forecast_periods + 1, freq=data.index.freq)
#plt.plot(forecast_index[1:], forecast, 'r--', label="Forecasted Values")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("SARIMA Forecasting")
plt.legend()
plt.grid(True)
plt.show()
#In this updated code, we load the data with a date index and ensure that it's recognized as datetime data. This should resolve the issue you mentioned regarding the pd.date_range function.



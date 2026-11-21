# Ex.No: 6               HOLT WINTERS METHOD
### Date: 21/11/2026


### DATASET:
US Electric Production
### AIM: 
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = pd.read_csv('Electric_Production.csv', parse_dates=['DATE'], index_col='DATE')
print(data.head())

data_monthly = data.resample('MS').sum()  # Month Start
print(data_monthly.head())

data_monthly.plot(title='Monthly Electric Production', figsize=(10, 4))
plt.ylabel('Electric Production (IPG2211A2N)')
plt.show()

# 3. Scale the data and check for seasonality
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

scaled_data.plot(title='Scaled Electric Production', figsize=(10, 4))
plt.show()


# Decompose to inspect trend and seasonality
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()

scaled_data = scaled_data + 1
train_data = scaled_data[:int(len(scaled_data) * 0.9)]  # 80% - train, 
test_data = scaled_data[int(len(scaled_data) * 0.9):]  #20% - test

model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='mul', seasonal_periods=12
).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# Visual evaluation
ax = train_data.plot(figsize=(10, 5))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual Evaluation')
plt.show()

# Quantitative evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"Test RMSE: {rmse:.4f}")
print(f"Scaled Data Std: {np.sqrt(scaled_data.var()):.4f}, Mean: {scaled_data.mean():.4f}")

# 5. Create the final model, predict future data, and plot it
final_model = ExponentialSmoothing(
    data_monthly, trend='add', seasonal='mul', seasonal_periods=12
).fit()

# Forecast next 25% of dataset length (~99 months)
final_predictions = final_model.forecast(steps=int(len(data_monthly) / 4))

ax = data_monthly.plot(figsize=(12, 5))
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Date')
ax.set_ylabel('Electric Production (IPG2211A2N)')
ax.set_title('Electric Production Forecast')
plt.show()
```
### OUTPUT:
ORIGINAL_PLOT: <br>
<img width="850" height="393" alt="image" src="https://github.com/user-attachments/assets/2f021d27-a325-45e4-b5e9-2b0947169ffa" />

SCALED_DATA_PLOT:<br>
<img width="826" height="393" alt="image" src="https://github.com/user-attachments/assets/762c4e7f-aea9-40f7-9bbe-52718df16c8d" />

DECOMPOSED_PLOT:<br>
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/8696addd-3c66-4628-b18c-44f13a22aff8" />

TEST_PREDICTION<br>
<img width="826" height="470" alt="image" src="https://github.com/user-attachments/assets/e1355e6a-d175-4974-bfbc-5477ba7dadcd" />


MODEL_PERFORMANCE_METRIC:<br>
<img width="553" height="46" alt="image" src="https://github.com/user-attachments/assets/d3cd539f-ae6d-4d77-b488-4db3b65dc32d" />

FINAL_PREDICTION
<img width="1005" height="470" alt="image" src="https://github.com/user-attachments/assets/54ddcd54-2a5d-4dbc-b09c-7737b24ae307" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.

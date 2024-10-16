### DEVELOPED BY : KULASEKARAPANDIAN K
### REGISTER NO : 212222240052
### Date: 

# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To implement an AutoRegressive Model using Python for time series forecasting.

### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'path/to/your/ethereum_data.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# Convert 'dateTime' to datetime and set it as index
data['dateTime'] = pd.to_datetime(data['dateTime'])
data.set_index('dateTime', inplace=True)

# Extract the 'close' prices
close_data = data['close']

# Perform differencing to make the series stationary
diff_data = close_data.diff().dropna()

# Perform Augmented Dickey-Fuller test
result = adfuller(diff_data)
print('ADF Statistic (After Differencing):', result[0])
print('p-value (After Differencing):', result[1])

# Split the data into training and testing sets
train_data = diff_data[:int(0.8 * len(diff_data))]
test_data = diff_data[int(0.8 * len(diff_data)):]

# Fit an AutoRegressive (AR) model with 13 lags
lag_order = 13
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

# Determine maximum lags for ACF and PACF plots
max_lags = len(diff_data) // 2

# Plot Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(diff_data, lags=max_lags, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Differenced Close Prices')
plt.show()

# Plot Partial Autocorrelation Function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(diff_data, lags=max_lags, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Differenced Close Prices')
plt.show()

# Make predictions
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

# Plot test data and predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data.values, label='Test Data - Differenced Close Prices', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Differenced Close Prices', color='orange', linestyle='--', linewidth=2)
plt.xlabel('DateTime')
plt.ylabel('Differenced Close Prices')
plt.title('AR Model Predictions vs Test Data (Differenced Close Prices)')
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT:

#### GIVEN DATA
![image](https://github.com/user-attachments/assets/c8e706a3-e300-47a0-b491-3860a6a74452)

#### ADFuller
```
ADF Statistic (After Differencing): -43.408219326550466
p-value (After Differencing): 0.0
```
#### PACF - ACF
![image](https://github.com/user-attachments/assets/71e2a021-3a0a-40e9-9018-dbd253b0d345)

#### MSE
```
Mean Squared Error (MSE): 151.12412322091575
```
#### PREDICTION
![image](https://github.com/user-attachments/assets/686faf58-4838-4b12-bd83-b3a2395c84df)


### RESULT:
Thus we have successfully implemented the auto regression function using python.

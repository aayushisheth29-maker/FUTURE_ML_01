import pandas as pd

# Load the sales demand dataset
file_path = "data/Daily_Demand_Forecasting_Orders.csv"

df = pd.read_csv(file_path, sep=";")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display the column names
print("\nColumn names:")
print(df.columns.tolist())

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Check data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistical summary
print("\nStatistical summary:")
print(df.describe())

import matplotlib.pyplot as plt

# Plot total orders
plt.figure(figsize=(10, 5))
plt.plot(df["Target (Total orders)"], marker="o")

plt.title("Daily Total Orders")
plt.xlabel("Record Number")
plt.ylabel("Total Orders")

plt.grid(True)
plt.show()

# Select features and target
# Create a time trend feature
df["Time_Index"] = range(len(df))
# Use time-based features for forecasting
X = df[
    [
        "Time_Index",
        "Week of the month (first week, second, third, fourth or fifth week",
        "Day of the week (Monday to Friday)"
    ]
]

y = df["Target (Total orders)"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)



# Split the data chronologically for forecasting
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)
print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nPredicted total orders:")
print(y_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Evaluation:")
print("Mean Absolute Error (MAE):", rf_mae)
print("Mean Squared Error (MSE):", rf_mse)
print("R² Score:", rf_r2)

# Visualise actual vs predicted demand

plt.figure(figsize=(10, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual Demand"
)

plt.plot(
    range(len(rf_pred)),
    rf_pred,
    marker="x",
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Total Orders")
plt.xlabel("Test Data Records")
plt.ylabel("Total Orders")
plt.legend()
plt.grid(True)

plt.show()

# Business Interpretation

print("\nBusiness Insights:")
print("1. The model can be used to analyse patterns in historical order demand.")
print("2. Actual and predicted demand can be compared to identify forecasting differences.")
print("3. Forecasting can help businesses plan resources according to expected demand.")
print("4. The current model shows room for improvement, as the R² score is negative.")


# Improved Random Forest Model

improved_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

# Train the improved model
improved_rf.fit(X_train, y_train)

# Make predictions
improved_pred = improved_rf.predict(X_test)

# Evaluate the improved model
improved_mae = mean_absolute_error(y_test, improved_pred)
improved_mse = mean_squared_error(y_test, improved_pred)
improved_r2 = r2_score(y_test, improved_pred)

print("\nImproved Random Forest Evaluation:")
print("Mean Absolute Error (MAE):", improved_mae)
print("Mean Squared Error (MSE):", improved_mse)
print("R² Score:", improved_r2)
# Final Forecast Visualisation

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values,
    marker="o",
    label="Actual Demand"
)

plt.plot(
    rf_pred,
    marker="x",
    label="Random Forest Forecast"
)

plt.title("Final Sales Demand Forecast")
plt.xlabel("Test Data Records")
plt.ylabel("Total Orders")
plt.legend()
plt.grid(True)

plt.show()

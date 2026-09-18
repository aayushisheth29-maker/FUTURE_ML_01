# FUTURE_ML_01 — Sales and Demand Forecasting

## Internship Task

Machine Learning Internship — Future Interns

### Task 1: Sales & Demand Forecasting for Businesses

This project focuses on analysing historical business order data and building machine learning models to forecast total order demand.

## Objectives

- Clean and explore historical order data
- Perform basic data analysis and visualisation
- Create time-based features for forecasting
- Train machine learning models
- Evaluate model performance using MAE, MSE, and R²
- Compare actual and predicted demand
- Generate business-friendly insights

## Dataset

Dataset: Daily Demand Forecasting Orders

The dataset contains 60 records and 13 original columns related to order demand, including week information, day of the week, different order categories, and total orders.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyCharm

## Machine Learning Models

### Linear Regression

Used as a baseline forecasting model.

### Random Forest Regressor

Used as a second model to capture non-linear patterns in the data.

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Results

The Random Forest model produced a lower MAE and MSE than the Linear Regression model on the test data.

However, the R² score remained negative, indicating that the current forecasting approach has limitations and requires further improvement for stronger predictive performance.

## Visualisations

The project includes:

- Daily Total Orders graph
- Actual vs Predicted Demand graph
- Final Sales Demand Forecast graph

## Business Insights

- Historical order data can be analysed to identify demand patterns.
- Actual and predicted demand can be compared to understand forecasting differences.
- Demand forecasting can support business resource planning.
- The current model provides a starting point but requires further improvement for more reliable forecasting.

## Project Structure

```text
FUTURE_ML_01/
│
├── data/
│   └── Daily_Demand_Forecasting_Orders.csv
│
├── task1_forecasting.py
│
└── README.md

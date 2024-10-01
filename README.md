# linear_regression
## my first model :D
# Linear Regression Model with Salary Prediction

## Project Overview
This project implements a simple **Linear Regression model** from scratch in Python to predict salaries based on years of experience. The dataset used for training and testing contains two features: `YearsExperience` (independent variable) and `Salary` (dependent variable).

The data preprocessing and splitting are done using libraries like `pandas`, `numpy`, and `scikit-learn`.

## Features
- Implements linear regression from scratch using `numpy`.
- Predicts `Salary` based on `YearsExperience`.
- Uses **gradient descent** to optimize model parameters.
- Calculates and outputs the **Root Mean Squared Error (RMSE)** to evaluate model performance.
- Includes **data visualization** to display the regression line and actual data points.

## Dataset
The dataset used in this project is a simple CSV file with the following structure:

| Unnamed: 0 | YearsExperience | Salary  |
|------------|-----------------|---------|
| 0          | 1.2             | 39344.0 |
| 1          | 1.4             | 46206.0 |
| 2          | 1.6             | 37732.0 |
| ...        | ...             | ...     |

The target variable is `Salary`, and the feature is `YearsExperience`.


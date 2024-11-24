# Car Sales Forecasting

📖 The goal of thos project is to predict car purchase amounts based on customer data and advertising expenditure. The model uses machine learning algorithms to forecast car sales by analyzing factors such as age, salary, and net worth of customers, as well as TV, radio, and newspaper advertising spend.

---

## 🗂️Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#Evaluation)
- [License](#license)

---

## Overview

This project uses machine learning techniques to predict car purchase amounts. The dataset consists of two main components:
    - Customer Data: Includes information such as age, annual salary, net worth, and car purchase amount.
    -Advertising Data: Includes the amount spent on TV, radio, and newspaper advertising for the car brand.

The goal is to build a model that predicts car purchase amounts based on customer and advertising features. The model uses Linear Regression and KMeans clustering for customer segmentation.

---

## Requirements

This project requires Python 3.x and the following libraries:
    - Python 3.x
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn


To install the required libraries, you can use the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🛠️Setup

1. Clone this repository to your local machine.

```bash
git clone https://github.com/your-username/car-sales-forecasting.git
```

2. Navigate into the project directory

```bash
cd car-sales-forecasting
```

3. Once you've set up the project and installed the requirements, you can run the script to train the model and make predictions by running:

```bash
python car_sales_forecasting.py
```

---

## Usage

This project includes a script (car_sales_forecasting.py) that performs the following

1. Data Preprocessing:

    Handles missing values and unnecessary columns.
    Scales numerical features using StandardScaler.
    Creates a new age group feature for customer segmentation.

2. Customer Segmentation:

    Applies KMeans clustering to group customers into clusters based on their age, salary, and net worth.
    Visualizes the segmentation by plotting a scatter plot of customer clusters.

3. Model Training:

    Trains a Linear Regression model to predict car purchase amounts using customer data (age, salary, net worth) and advertising data (TV, radio, newspaper).

4. Model Evaluation:

    Evaluates model performance using Mean Squared Error (MSE).
    Visualizes the relationship between actual and predicted car purchase amounts.

5. Model Prediction:

    Allows you to input new advertising spend values and predict car sales for the given input.
---

## Model Evaluation
The model's performance is evaluated using Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values. A lower MSE indicates a more accurate model.
    - Mean Squared Error (MSE): Provides a measure of how far off the predictions are from the actual car purchase amounts.
    - R-Squared Score: Shows how well the model's predictions fit the actual data.
---

## 📜 License

This project is licensed under the MIT License. See the LICENSE tab on the top for more details.

---
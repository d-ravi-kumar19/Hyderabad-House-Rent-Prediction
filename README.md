# Hyderabad House Rent Prediction

## Overview
This repository contains a project focused on predicting house rents in Hyderabad, India. The project involves data cleaning, exploratory data analysis, feature engineering, and the development of a machine learning model for rent prediction. The model is deployed in a Streamlit app (`app.py`) for user-friendly input and prediction.

## Project Structure
- **.ipynb_checkpoints**: Directory containing Jupyter Notebook checkpoints.
- **data**: Directory containing datasets used for cleaning (`cleaned_data.csv`) and the original data (`Hyd_house_rent_data.csv`).
- **app.py**: Streamlit app script for taking user input and predicting house rents.
- **house_rent_prediction.ipynb**: Jupyter Notebook with the code for data cleaning, exploration, feature engineering, and model development.
- **hyd_house_rent_prices.pkl**: Pickle file containing the trained Linear Regression model for house rent prediction.

## Usage
1. **Jupyter Notebook**: Use `house_rent_prediction.ipynb` for detailed exploration, analysis, and model development.
2. **Streamlit App**: Run the Streamlit app using the command `streamlit run app.py` and open the provided URL in your web browser to input values and get rent predictions.

## Dataset
- **Original Data**: `Hyd_house_rent_data.csv` - The raw dataset containing information about various factors influencing house rents in Hyderabad.
- **Cleaned Data**: `cleaned_data.csv` - The cleaned dataset used for model training and testing.

## Model
The machine learning model, based on Linear Regression, is trained on the cleaned dataset (`cleaned_data.csv`). The trained model is saved as `hyd_house_rent_prices.pkl` for later use in making predictions.

## Instructions
1. Explore the Jupyter Notebook (`house_rent_prediction.ipynb`) for an in-depth understanding of the project.
2. Run the Streamlit app (`app.py`) for a user-friendly interface to input values and get rent predictions.

Feel free to contribute, report issues, or suggest improvements!

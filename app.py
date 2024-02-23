import streamlit as st
import pandas as pd
import numpy as np
import pickle


cleaned_df = pd.read_csv('data/cleaned_data.csv')
X = cleaned_df.drop(['rent_amount','localityId'], axis=1)

with open('hyd_house_rent_prices.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

def reverse_type_bhk_encoding(value):
    d = {0.5: "RK1", 1: 'BHK1', 2: 'BHK2', 3: 'BHK3', 4: 'BHK4', 5: 'BHK4PLUS'}
    return d.get(value)

def reverse_parking_encoding(value):
    d = {0: "None", 1: '2 wheeler', 2: '4 wheeler', 3: 'Both'}
    return d.get(value)

def reverse_lift_encoding(value):
    d = {0: "No", 1: 'Yes'}
    return d.get(value)

def reverse_furnishing_encoding(value):
    d = {0.5: "Semi funrnished", 1: 'Fully furnished', 0: 'Unfurnished'}
    return d.get(value)

def predict_price(localityId, selected_type_bhk, selected_balconies, bathroom, parking, lift, furnishingDesc, property_size, maintenance):
    loc_index = np.where(X.columns == localityId)[0]

    if loc_index:
        loc_index = loc_index[0]
        x = np.zeros(len(X.columns))
        x[0] = selected_type_bhk
        x[1] = selected_balconies
        x[2] = bathroom
        x[3] = parking
        x[4] = lift
        x[5] = furnishingDesc
        x[6] = property_size
        x[7] = maintenance

        if loc_index >= 0:
            x[loc_index] = 1

        return best_model.predict([x])[0]
    else:
        return "LocalityId not found in training data."


st.title('Hyderabad House Rent Prediction')
locality_options = cleaned_df['localityId'].unique()
localityId = st.selectbox('Locality', locality_options) # localityId

type_bhk_options = cleaned_df['type_bhk'].unique()   
selected_type_bhk = st.selectbox('Type BHK', type_bhk_options, format_func=reverse_type_bhk_encoding)  # type_bhk

selected_balconies = st.number_input('Balconies', min_value=0, value=1)                                 # bolconies

bathroom = st.number_input('Bathroom', min_value=1, value=1, max_value=5)                               # bathrooms

parking_options = cleaned_df['parking'].unique()                                                        
parking = st.selectbox('Parking', parking_options, format_func=reverse_parking_encoding)                # parking

lift_options = cleaned_df['lift'].unique()
lift = st.selectbox('Lift', lift_options, format_func=reverse_lift_encoding)                            # lift

furnishing_options = cleaned_df['furnishingDesc'].unique()
furnishingDesc = st.selectbox('Furnishing', furnishing_options, format_func=reverse_furnishing_encoding)  # furnishing

property_size = st.number_input('Property Size', min_value=1, value=1000)                                   # property size

maintenance = st.number_input('Maintenance', min_value=1000, value=1000)                                    # maintenance

if st.button('Predict Rent'):
    if localityId in X.columns:
        prediction_result = predict_price(localityId, selected_type_bhk, selected_balconies, bathroom,
                                    parking, lift, furnishingDesc, property_size, maintenance)
        st.success(f'Predicted Rent:  {float(prediction_result):.2f} INR')
    else:
        st.error(f'Selected Locality "{localityId}" not found in training data.')




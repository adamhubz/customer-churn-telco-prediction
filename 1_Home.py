import pandas as pd
import streamlit as st
import numpy as np
import pickle
from scipy.stats.mstats import winsorize

st.set_page_config(page_title = 'Customer Churn Telco Prediction', page_icon=':iphone:', layout='wide')
st.write("""
# Customer Churn Telco Prediction
Menggunakan machine learning untuk memprediksi apakah customer menjadi churn atau tidak.
""")

def user_input_features():
    state = st.selectbox('Which state are you from?', ['OH', 'NJ', 'OK', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY', 'ID', 'VA',
                                                        'TX', 'FL', 'CO', 'AZ', 'SC', 'WY', 'HI', 'NH', 'AK', 'GA', 'MD', 'AR', 'WI', 'OR',
                                                        'MI', 'DE', 'UT', 'CA', 'SD', 'NC', 'WA', 'MN', 'NM', 'NV', 'DC', 'VT', 'KY', 'ME',
                                                        'MS', 'AL', 'NE', 'KS', 'TN', 'IL', 'PA', 'CT', 'ND'])
    account_length = st.number_input('How long have you been a telco customer? (in Month)', 0, 100000)
    area_code = st.selectbox('What is your area code?', ['area_code_415', 'area_code_408', 'area_code_510'])
    international_plan = st.selectbox('do you have international plan service?', ['yes', 'no'])
    voice_mail_plan = st.selectbox('do you have voice mail plan service?', ['yes', 'no'])
    number_vmail_messages =  st.number_input('How many total voice mail messages?', 0, 1000000000)
    total_day_minutes = st.number_input('How many total day minutes?', 0, 1000000000)
    total_day_calls = st.number_input('How many total day calls?', 0, 1000000000)
    total_day_charge = st.number_input('How many total day charge?', 0, 1000000000)
    total_eve_minutes = st.number_input('How many total eve minutes?', 0, 1000000000)
    total_eve_calls = st.number_input('How many total eve calls?', 0, 1000000000)
    total_eve_charge = st.number_input('How many total eve charge?', 0, 1000000000)
    total_night_minutes = st.number_input('How many total night minutes?', 0, 1000000000)
    total_night_calls = st.number_input('How many total night calls?', 0, 1000000000)
    total_night_charge = st.number_input('How many total night charge?', 0, 1000000000)
    total_intl_minutes = st.number_input('How many total international minutes?', 0, 1000000000)
    total_intl_calls = st.number_input('How many total international calls?', 0, 1000000000)
    total_intl_charge = st.number_input('How many total international charge?', 0, 1000000000)
    number_customer_service_calls = st.number_input('How many customer service have you called?', 0, 1000000000)
    data = {'state': state,
            'account_length': account_length,
            'area_code': area_code,
            'international_plan': international_plan,
            'voice_mail_plan': voice_mail_plan,
            'number_vmail_messages': number_vmail_messages,
            'total_day_minutes': total_day_minutes,
            'total_day_calls': total_day_calls,
            'total_day_charge': total_day_charge,
            'total_eve_minutes': total_eve_minutes,
            'total_eve_calls': total_eve_calls,
            'total_eve_charge': total_eve_charge,
            'total_night_minutes': total_night_minutes,
            'total_night_calls': total_night_calls,
            'total_night_charge': total_night_charge,
            'total_intl_minutes': total_intl_minutes,
            'total_intl_calls': total_intl_calls,
            'total_intl_charge': total_intl_charge,
            'number_customer_service_calls': number_customer_service_calls}
    features = pd.DataFrame(data, index = [0])
    return features

input_df = user_input_features()

train = pd.read_csv('datasets/train.csv')

def predict():
    X_test = input_df.copy()

    # Handle Outliers pada kolom normal distribution dengan winsorize pada data train
    for v in ['account_length', 'total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 
            'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_charge']:
        train['trans_'+v] = winsorize(train[v],limits=[.01,.99])

    # Handle Outliers pada kolom normal distribution dengan winsorize
    for v in ['account_length', 'total_day_calls', 'total_day_charge', 'total_eve_calls', 'total_eve_charge', 'total_night_calls', 'total_night_charge', 'total_intl_charge']:
        X_test.loc[(X_test[v] <= train['trans_'+v].min()), 'trans_'+v] = train['trans_'+v].min()
        X_test.loc[(X_test[v] >= train['trans_'+v].max()), 'trans_'+v] = train['trans_'+v].max()

    # Handle Outliers pada kolom skewed dan bimodal distribution dengan log transformation
    for l in ['number_vmail_messages', 'total_intl_calls', 'number_customer_service_calls']:
        X_test['trans_'+l] = np.log(X_test[l]+1)

    # Encode Label
    for col in ['international_plan', 'voice_mail_plan']:
        X_test[col] = X_test[col].apply(lambda x: 1 if x == 'yes' else 0)

    # One Hot Encoding
    for cat in ['state', 'area_code']:
        onehots = pd.get_dummies(train[cat], prefix=cat, drop_first = True)
        onehots.loc[0, :] = 0
        val = onehots.head(1)
        if X_test[cat].values[0] in val.columns:
            val[cat+'_'+X_test[cat].values] = 1
        X_test = X_test.join(onehots)

    # drop kolom state, area_code
    X_test.drop(['state', 'area_code'], axis = 1, inplace = True)

    # Drop Kolom yang tidak digunakan
    X_test.drop(columns = ['account_length', 'total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 
                'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_charge', 'number_vmail_messages', 'total_intl_calls', 'number_customer_service_calls',], inplace = True)

    # Scale Test set
    load_scaler = pickle.load(open('scaler_clf.pkl', 'rb'))
    X_test_scaled = load_scaler.transform(X_test)

    # Transform the scaled test features using pca
    load_pca = pickle.load(open('pca_clf.pkl', 'rb'))
    X_scaled_pca = load_pca.transform(X_test_scaled)

    # Predict Scaled pca test data menggunakan machine learning random forest yang telah di training
    load_model = pickle.load(open('model_clf.pkl', 'rb'))
    prediction = load_model.predict(X_scaled_pca)[0]
    prediction_prob = load_model.predict_proba(X_scaled_pca)

    if prediction == 1:
        st.warning(f"""Selamat kamu mendapatkan penawaran khusus, ***HANYA UNTUK KAMU!***
                    Kamu mendapatkan pesan ini karena kamu diprediksi akan meninggalkan layanan ini dengan peluang {prediction_prob[0, prediction] * 100}%""", icon="⚠️")
    else:
        st.success(f'Kamu tidak diprediksi akan meninggalkan layanan ini dengan peluang {prediction_prob[0, prediction] * 100}%')

st.button('Predict', on_click=predict)





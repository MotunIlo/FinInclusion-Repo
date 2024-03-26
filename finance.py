import streamlit as st
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
import joblib

st.markdown("<h1 style = 'color: #008DDA; text-align: center; font-family: Helvetica'>FINANCIAL INCLUSION DATASET</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFB000; text-align: center; font-family: Brush Script MT'> Built By: ILO MOTUNRAYO</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (3).png', use_column_width=True)

st.header('Project Background Information', divider = True)
st.write('The primary objective of the predictive model is to leverage machine learning algorithms to analyze demographic, socio-economic, and behavioral data, thereby accurately identifying individuals who are most likely to possess or utilize a bank account, with the aim of informing targeted outreach strategies, guiding policymakers in allocating resources effectively, empowering financial institutions to tailor products and services to diverse customer needs, and ultimately contributing to the advancement of comprehensive financial inclusion efforts on a broader scale.')

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('Financial_inclusion_dataset.csv')
st.dataframe(data.drop('uniqueid', axis = 1))

st.sidebar.image('pngwing.com (4).png', width = 300, caption = 'Welcome User')
st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Input User Image 
# st.sidebar.image('pngwing.com-15.png', caption = 'Welcome User')

# Apply space in the sidebar 
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)


# Declare user Input variables 
st.sidebar.subheader('Input Variables', divider= True)
age = st.sidebar.number_input('age_of_respondent', data['age_of_respondent'].min(), data['age_of_respondent'].max())
house_hold = st.sidebar.number_input('household_size', data['household_size'].min(), data['household_size'].max())
job = st.sidebar.selectbox('job_type', data['job_type'].unique())
edu_level = st.sidebar.selectbox('education_level', data['education_level'].unique())
mar_status = st.sidebar.selectbox('marital_status', data['marital_status'].unique())
count_ry = st.sidebar.selectbox('country', data['country'].unique())
location = st.sidebar.selectbox('location_type', data['location_type'].unique())
rlship = st.sidebar.selectbox('relationship_with_head', data['relationship_with_head'].unique())

# display the users input
input_var = pd.DataFrame()
input_var['age_of_respondent'] = [age]
input_var['household_size'] = [house_hold]
input_var['job_type'] = [job]
input_var['education_level'] = [edu_level]
input_var['marital_status'] = [mar_status]
input_var['country'] = [count_ry]
input_var['location_type'] = [location]
input_var['relationship_with_head'] = [rlship]



st.markdown("<br>", unsafe_allow_html= True)
# display the users input variable 
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)

job = joblib.load('job_type_encoder.pkl')
edu_level = joblib.load('education_level_encoder.pkl')
mar_status = joblib.load('marital_status_encoder.pkl')
count_ry = joblib.load('country_encoder.pkl')
location = joblib.load('location_type_encoder.pkl')
rlship = joblib.load('relationship_with_head_encoder.pkl')


# transform the users input with the imported scalers 
input_var['job_type'] = job.transform(input_var[['job_type']])
input_var['education_level'] =  edu_level.transform(input_var[['education_level']])
input_var['marital_status'] = mar_status.transform(input_var[['marital_status']])
input_var['country'] = count_ry.transform(input_var[['country']])
input_var['location_type'] = location.transform(input_var[['location_type']])
input_var['relationship_with_head'] = rlship.transform(input_var[['relationship_with_head']])


model = joblib.load('FinInclusModel.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

if st.button('Predict'):
    if predicted == 0:
        st.error('Unlikely to have a bank account.')
        st.image('Red_Prohibited_sign_No_icon_warning_or_stop_symbol_safety_danger_isolated_vector_illustration-removebg-preview.png', width = 200)
    else:
        st.success('Likely to have a bank account.')
        st.image('pngused.png', width = 200)
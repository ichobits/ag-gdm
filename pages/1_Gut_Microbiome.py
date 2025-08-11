import streamlit as st
from streamlit_shap import st_shap
import autogluon.core as ag
from autogluon.tabular import TabularPredictor
import pandas as pd
import shap

st.set_page_config(page_title="Gut MciroBiome Prediction", layout="wide")

feature_names = [
    'ASV_409',
    'ASV_2560',
    'ASV_574',
    'ASV_414',
    'ASV_710',
    'ASV_290',
    'ASV_728',
    'ASV_642',
    'ASV_308',
    'ASV_283',
    'ASV_1988',
    'ASV_252',
    'ASV_112',
    'ASV_188',
    'ASV_357',
    'ASV_108',
    'ASV_371',
    'ASV_139',
    ]
asv_famliy =pd.read_csv('./pages/asv_filter_family.csv',index_col=0)

default_values = {
    'ASV_409': 37.0, 
    'ASV_2560': 0.0, 
    'ASV_574': 0.0, 
    'ASV_414': 0.0, 
    'ASV_710': 0.0, 
    'ASV_290': 42.0, 
    'ASV_728': 0.0, 
    'ASV_642': 11.0, 
    'ASV_308': 0.0,
    'ASV_283': 0.0, 
    'ASV_1988':0.0, 
    'ASV_252':0.0, 
    'ASV_112':153.0, 
    'ASV_188':0.0,
    'ASV_357':0.0,
    'ASV_108':47.0,
    'ASV_371':0.0,
    'ASV_139':0.0,
    }

class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, as_multiclass=False)


predictor = TabularPredictor.load("./AutogluonModels/ag-20240711_092249/") 
predictor.set_model_best('CatBoost_r143_BAG_L1')


med_data = {
    'ASV_409': [12.0],
    'ASV_2560': [0.0],
    'ASV_574': [0.0],
    'ASV_414': [0.0],
    'ASV_710': [0.0],
    'ASV_290': [28.5],
    'ASV_728': [0.0],
    'ASV_642': [0.0],
    'ASV_308': [8.5],
    'ASV_283': [0.0],
    'ASV_1988':[0.0],
    'ASV_252':[0.0],
    'ASV_112':[0.0],
    'ASV_188':[0.0],
    'ASV_357':[0.0],
    'ASV_108':[0.0],
    'ASV_371':[0.0],
    'ASV_139':[55.5],
    }

med = pd.DataFrame(med_data)


st.title("Binary Prediction For GDM Microbiome")
st.write("This app allows you to predict the class of an input sample using an Binary model.")
st.markdown("### The features of the GDM Microbiome dataset are as follows:")
st.dataframe(asv_famliy,hide_index=True)

st.sidebar.subheader("Input Features")


st.subheader("Please input the values of GDM Microbiome features:")

user_data = {}
col1, col2, col3 = st.columns(3)
with col1:
    user_data['ASV_409'] = st.number_input("ASV_409 value", value=default_values['ASV_409'])
    user_data['ASV_2560'] = st.number_input("ASV_2560 value", value=default_values['ASV_2560'])
    user_data['ASV_574'] = st.number_input("ASV_574 value", value=default_values['ASV_574'])
    user_data['ASV_414'] = st.number_input("ASV_414 value", value=default_values['ASV_414'])
    user_data['ASV_710'] = st.number_input("ASV_710 value", value=default_values['ASV_710'])
    user_data['ASV_290'] = st.number_input("ASV_290 value", value=default_values['ASV_290'])

with col2:
    
    user_data['ASV_728'] = st.number_input("ASV_728 value", value=default_values['ASV_728'])
    user_data['ASV_642'] = st.number_input("ASV_642 value", value=default_values['ASV_642'])
    user_data['ASV_308'] = st.number_input("ASV_308 value", value=default_values['ASV_308'])
    user_data['ASV_283'] = st.number_input("ASV_283 value", value=default_values['ASV_283'])
    user_data['ASV_1988'] = st.number_input("ASV_1988 value", value=default_values['ASV_1988'])
    user_data['ASV_252'] = st.number_input("ASV_252 value", value=default_values['ASV_252'])

with col3:
        
    user_data['ASV_112'] = st.number_input("ASV_112 value", value=default_values['ASV_112'])
    user_data['ASV_188'] = st.number_input("ASV_188 value", value=default_values['ASV_188'])
    user_data['ASV_357'] = st.number_input("ASV_357 value", value=default_values['ASV_357'])
    user_data['ASV_108'] = st.number_input("ASV_108 value", value=default_values['ASV_108'])
    user_data['ASV_371'] = st.number_input("ASV_371 value", value=default_values['ASV_371'])
    user_data['ASV_139'] = st.number_input("ASV_139 value", value=default_values['ASV_139'])

matching_rows = asv_famliy[asv_famliy['ASV_ID'].isin(user_data.keys())]
matching_rows_c=matching_rows.copy()
matching_rows_c['value'] = matching_rows_c['ASV_ID'].map(user_data)
user_data_input = matching_rows_c[['ASV_ID','genus','species','value']]
input_ml=pd.DataFrame(user_data, index=[0]) 

med_match = med.columns[med.columns.isin(input_ml.columns)]
med_same_input = med[med_match]

st.markdown("#### The Values for user_data_input:")
st.dataframe(user_data_input)

st.markdown("#### Classification meaning:")
gdm_group = pd.DataFrame({'N Group':[0], 'GDM Group':[1]})
st.dataframe(gdm_group, hide_index=True)

if st.button("Predict"):
    prediction = predictor.predict(input_ml)  
    
    prediction_proba = predictor.predict_proba(input_ml)
    
    st.write(f"Predicted class: {prediction[0]}")
    
    st.write("Class probabilities:")
    st.dataframe(prediction_proba, hide_index=True)
    
    ag_wrapper = AutogluonWrapper(predictor, feature_names)
    explainer  = shap.KernelExplainer(ag_wrapper.predict_binary_prob, med)
    single_datapoint = input_ml
    shap_values_single = explainer.shap_values(single_datapoint)
    
    if prediction[0] == 1:
        st.markdown(
        """         
            - According to our model, you have a **high risk** of GDM (Gestational Diabetes Mellitus).
            - Please consult a doctor for further evaluation and advice.          
            - It's advised to consult with your healthcare provider for further evaluation and possible intervention.
         """
         )
    else:
        st.markdown(
            """
            - According to our model, you have a **low risk** of GDM (Gestational Diabetes Mellitus).
            - Please consult a doctor for further evaluation and advice.
            - It's advised to consult with your healthcare provider for further evaluation and possible intervention.
            """
            )

    
    st.write("### SHAP Force Plot")
    shap.initjs()  
    if prediction[0] == 1:
        st.write("This 1 class")

        st.markdown("""
            #### Key Visual Elements
        - Red (positive): Features that push the prediction higher.
        - Blue (negative): Features that pull the prediction lower.
        - Length of the arrow: Indicates the magnitude of the feature's impact on the prediction.
        - Base value: The starting point before the features were considered.
        """)
        st_shap(
            shap.force_plot(explainer.expected_value, shap_values_single, single_datapoint),
            height=400,
            width=1000,
        )

    else:
        st.write("This 0 class")
        st.markdown("""
            #### Key Visual Elements
        - Red (positive): Features that push the prediction higher.
        - Blue (negative): Features that pull the prediction lower.
        - Length of the arrow: Indicates the magnitude of the feature's impact on the prediction.
        - Base value: The starting point before the features were considered.
        """)
        st_shap(
            shap.force_plot(explainer.expected_value, shap_values_single, single_datapoint),
            height=400,
            width=1000,
        )
import streamlit as st
from streamlit_shap import st_shap
import autogluon.core as ag
from autogluon.tabular import TabularPredictor
import pandas as pd
import shap

st.set_page_config(page_title="Gut Metabolome Prediction", layout="wide")

feature_names = [
    'm_98',
    'm_904',
    'm_1214',
    'm_3304',
    'm_4667',
    'm_5484',
    'm_6538',
    'm_7017',
    'm_7580',
    'm_8683',
]

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

predictor = TabularPredictor.load("./AutogluonModels/ag-20250326_034816") 
predictor.set_model_best('CatBoost')



data = {
    'MID': ['m_98', 'm_904', 'm_1214', 'm_3304', 'm_4667', 'm_5484', 'm_6538', 'm_7017', 'm_7580', 'm_8683'], 
    'Metabolites': [
        "C17 Sphinganine", 
        "2-Hydroxypropyl 2-Isopropyl-5-Methylcyclohexyl Carbonate", 
        "Canarigenin 3-[Glucosyl-(1->4)-6-Deoxy-Alloside]", 
        "16-Oxoandrostenediol", 
        "3-O-Alpha-D-Glucopyranuronosyl-D-Xylose", 
        "Phe Glu Phe", 
        "N-Phosphocreatinate(2-)", 
        "1,2-Dibutyrin", 
        "Potassium Gluconate",
        "5-Amino-1,3,4-Thiadiazole-2-Thiol"
    ],
    'df_val': [81124.023461, 6493.552112, 362.359533, 259.609045, 3386.993392, 11237.983930, 3805.757181, 1999.191648, 1575.895645, 2279.901048]
}

df = pd.DataFrame(data)

st.title("Binary Prediction For GDM Metabolites")
st.write("This app allows you to predict the class of an input sample using an Binary model.")


st.sidebar.header("Input Features")

features = {}

for index, row in df.iterrows():
    user_input = st.sidebar.number_input(
        label=row['Metabolites'],  
        min_value=-1000000000000.0, 
        max_value= 1000000000000.0, 
        value=row['df_val'], 
        step=0.1
    )
    features[row['MID']] = user_input

input_data = pd.DataFrame([features])
input_data.columns = df['Metabolites']
transposed_data = input_data.T.reset_index()
transposed_data.columns = ['Metabolites', 'YourData']

input_ml = pd.DataFrame([transposed_data['YourData'].values], columns=df['MID'])
input_ml.reset_index(drop=True, inplace=True)
input_ml_for_show = pd.concat([df['MID'], transposed_data], axis=1)

med_data = {
    'm_6538': [3805.757181],
    'm_3304': [259.609045],
    'm_1214': [362.359533],
    'm_904': [6493.552112],
    'm_7580': [1575.895645],
    'm_7017': [1999.191648],
    'm_8683': [2279.901048],
    'm_4667': [3386.993392],
    'm_98': [81124.023461],
    'm_5484': [11237.983930],
}



med = pd.DataFrame(med_data)


st.markdown("#### User input features valuse:")
st.dataframe(input_ml_for_show, hide_index=True)



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
        advice=("According to our model, you have a high risk of (Gestational Diabetes Mellitus). \n" 
                "Please consult a doctor for further evaluation and advice.\n"           
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention.")
    else:
        advice=("According to our model, you have a low risk of heart disease. "
            "Please consult a doctor for further evaluation and advice."
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention.")

    st.markdown(advice)
    
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
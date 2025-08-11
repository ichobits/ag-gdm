import streamlit as st

st.set_page_config(
    page_title="GDM Prediction",
)

st.write("# Welcome to GDM Prediction!")

st.sidebar.success("Select a APP above.")

st.markdown(
    """
    This program is used for machine learning to predict 
    gut microbiome and metabolome of GDM (Gestational Diabetes Mellitus).

    ### Introduction
    We used machine learning to construct a binary model of GDM gut microbes and metabolites. Finally, we constructed a prediction model based on Catboost. The model of GDM gut microbes requires 18 feature values. The model of GDM metabolites requires 10 feature values. Fill in the data of the corresponding feature values to obtain the prediction probability of GDM.

    ### Paper citation
    - Applying machine learning to reveal the gut microbiome and metabolome of gestational diabetes mellitus
"""
)

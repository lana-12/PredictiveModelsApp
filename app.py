
import streamlit as st
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression2 import regression2_page


st.set_page_config(
    page_title="ProjectDataScience",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

if type_data == "Regression":
    regression2_page()
elif type_data == "Classification":
    classification_page()
elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")
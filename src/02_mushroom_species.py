from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('models/mushroom_species_model')

def predict(model, input_df):
    predictions_data = predict_model(estimator=model, data=input_df)
    predictions = predictions_data['prediction_label'][0]
    return predictions

def run():
    from PIL import Image
    mushroom_header = Image.open('src/statics/img/mushroom_header.png')
    mushroom = Image.open('src/statics/img/mushroom.jpg')
    st.image(mushroom_header, width=700)
    
    
    st.sidebar.info('This app is created to predict mushroom species')
    st.sidebar.image(mushroom, width=300) 

    st.title("Predicting Mushroom Species")

    cap_shape = st.selectbox("Cap Shape", ['convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical'])
    cap_surface = st.selectbox("Cap Surface", ['smooth', 'scaly', 'fibrous', 'grooves'])
    cap_color = st.selectbox("Cap Color", ['brown', 'yellow', 'white', 'gray', 'red', 'pink', 'buff', 'purple', 'cinnamon', 'green'])
    bruises = st.selectbox("Bruises", ['bruises', 'no'])
    odor = st.selectbox("Odor", ['pungent', 'almond', 'anise', 'none', 'foul', 'creosote', 'fishy', 'spicy', 'musty'])
    gill_attachment = st.selectbox("Gill Attachment", ['free', 'attached'])
    gill_spacing = st.selectbox("Gill Spacing", ['close', 'crowded'])
    gill_size = st.selectbox("Gill Size", ['narrow', 'broad'])
    gill_color = st.selectbox("Gill Color", ['black', 'brown', 'gray', 'pink', 'white', 'chocolate', 'purple', 'red', 'buff', 'green', 'yellow', 'orange'])
    stalk_shape = st.selectbox("Stalk Shape", ['enlarging', 'tapering'])
    stalk_root = st.selectbox("Stalk Root", ['equal', 'club', 'bulbous', 'rooted'])
    stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", ['smooth', 'fibrous', 'silky', 'scaly'])
    stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", ['smooth', 'fibrous', 'scaly', 'silky'])
    stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", ['white', 'gray', 'pink', 'brown', 'buff', 'red', 'orange', 'cinnamon', 'yellow'])
    stalk_color_below_ring = st.selectbox("Stalk Color Below Ring", ['white', 'pink', 'gray', 'buff', 'brown', 'red', 'yellow', 'orange', 'cinnamon'])
    veil_type = st.selectbox("Veil Type", ['partial'])
    veil_color = st.selectbox("Veil Color", ['white', 'brown', 'orange', 'yellow'])
    ring_number = st.selectbox("Ring Number", ['none','one', 'two'])
    ring_type = st.selectbox("Ring Type", ['pendant', 'evanescent', 'large', 'flaring', 'none'])
    spore_print_color = st.selectbox("Spore Print Color", ['black', 'brown', 'purple', 'chocolate', 'white', 'green', 'orange', 'yellow', 'buff'])
    population = st.selectbox("Population", ['scattered', 'numerous', 'abundant', 'several', 'solitary', 'clustered'])
    habitat = st.selectbox("Habitat", ['urban', 'grasses', 'meadows', 'woods', 'path', 'leaves'])

    input_dict = {
        'cap-shape': [cap_shape],
        'cap-surface': [cap_surface],
        'cap-color': [cap_color],
        'bruises': [bruises],
        'odor': [odor],
        'gill-attachment': [gill_attachment],
        'gill-spacing': [gill_spacing],
        'gill-size': [gill_size],
        'gill-color': [gill_color],
        'stalk-shape': [stalk_shape],
        'stalk-root': [stalk_root],
        'stalk-surface-above-ring': [stalk_surface_above_ring],
        'stalk-surface-below-ring': [stalk_surface_below_ring],
        'stalk-color-above-ring': [stalk_color_above_ring],
        'stalk-color-below-ring': [stalk_color_below_ring],
        'veil-type': [veil_type],
        'veil-color': [veil_color],
        'ring-number': [ring_number],
        'ring-type': [ring_type],
        'spore-print-color': [spore_print_color],
        'population': [population],
        'habitat': [habitat]
    }

    input_df = pd.DataFrame.from_dict(input_dict)

    # Prediction button
    if st.button("Predict"):
        prediction = predict(model, input_df)
        st.success(f"The predicted mushroom species is {prediction}")
    else:
        st.write("Click the button to get a prediction.")

if __name__ == "__main__":
    run() 
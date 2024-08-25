from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mushroom Prediction", page_icon="🍄")

st.markdown("# Mushroom Edibility Prediction 🍄")
st.sidebar.header("Mushroom Edibility Prediction")
model = load_model('models/mushroom_species_model')

def predict(model, input_df):
    predictions_data = predict_model(estimator=model, data=input_df)
    predictions = predictions_data['prediction_label'][0]
    score = predictions_data['prediction_score'][0]
    return predictions

def run():
    from PIL import Image
    mushroom_header = Image.open('src/statics/img/mushroom_header.png')
    st.image(mushroom_header, width=700)

    
    st.sidebar.info('This app is created to predict whether a mushroom is edible or poisonous based on its features. Please adjust the features accordingly.')
    st.sidebar.text('Done by: Lan (221520B)')

    st.info("Please select the features of the mushroom to predict its species.",icon="ℹ️")

    st.subheader("Cap Features")
    cap_shape = st.selectbox("Cap Shape", ('convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical'), placeholder="Choose an option", index=None)
    cap_surface = st.selectbox("Cap Surface", ('smooth', 'scaly', 'fibrous', 'grooves'), placeholder="Choose an option", index=None)
    cap_color = st.selectbox("Cap Color", ('brown', 'yellow', 'white', 'gray', 'red', 'pink', 'buff', 'purple', 'cinnamon', 'green'), placeholder="Choose an option", index=None)
    st.divider()
    st.subheader("Bruises and Odor")
    bruises = st.selectbox("Bruises", ('bruises', 'no'), placeholder="Choose an option", index=None)
    odor = st.selectbox("Odor", ('pungent', 'almond', 'anise', 'none', 'foul', 'creosote', 'fishy', 'spicy', 'musty'), placeholder="Choose an option", index=None)
    st.divider()
    st.subheader("Gill Features")
    gill_attachment = st.selectbox("Gill Attachment", ('free', 'attached'), placeholder="Choose an option", index=None)
    gill_spacing = st.selectbox("Gill Spacing", ('close', 'crowded'), placeholder="Choose an option", index=None)
    gill_size = st.selectbox("Gill Size", ('narrow', 'broad'), placeholder="Choose an option", index=None)
    gill_color = st.selectbox("Gill Color", ('black', 'brown', 'gray', 'pink', 'white', 'chocolate', 'purple', 'red', 'buff', 'green', 'yellow', 'orange'), placeholder="Choose an option", index=None)
    st.divider()
    st.subheader("Stalk Features")
    stalk_shape = st.selectbox("Stalk Shape", ('enlarging', 'tapering'), placeholder="Choose an option", index=None)
    stalk_root = st.selectbox("Stalk Root", ('equal', 'club', 'bulbous', 'rooted'), placeholder="Choose an option", index=None)
    stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", ('smooth', 'fibrous', 'silky', 'scaly'), placeholder="Choose an option", index=None)
    stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", ('smooth', 'fibrous', 'scaly', 'silky'), placeholder="Choose an option", index=None)
    stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", ('white', 'gray', 'pink', 'brown', 'buff', 'red', 'orange', 'cinnamon', 'yellow'), placeholder="Choose an option", index=None)
    stalk_color_below_ring = st.selectbox("Stalk Color Below Ring", ('white', 'pink', 'gray', 'buff', 'brown', 'red', 'yellow', 'orange', 'cinnamon'), placeholder="Choose an option", index=None)
    st.divider()
    st.subheader("Veil and Ring Features")
    veil_type = st.selectbox("Veil Type", ['partial'])
    veil_color = st.selectbox("Veil Color", ('white', 'brown', 'orange', 'yellow'), placeholder="Choose an option", index=None)
    ring_number = st.selectbox("Ring Number", ('none','one', 'two'), placeholder="Choose an option", index=None)
    ring_type = st.selectbox("Ring Type", ('pendant', 'evanescent', 'large', 'flaring', 'none'), placeholder="Choose an option", index=None)

    st.subheader("Spore Print, Population, and Habitat Features")
    spore_print_color = st.selectbox("Spore Print Color", ('black', 'brown', 'purple', 'chocolate', 'white', 'green', 'orange', 'yellow', 'buff'), placeholder="Choose an option", index=None)
    population = st.selectbox("Population", ('scattered', 'numerous', 'abundant', 'several', 'solitary', 'clustered'), placeholder="Choose an option", index=None)
    habitat = st.selectbox("Habitat", ('urban', 'grasses', 'meadows', 'woods', 'path', 'leaves'), placeholder="Choose an option", index=None)

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
        if input_df.isnull().values.any():
            blank_fields = input_df.columns[input_df.isnull().any()].tolist()
            st.error(f"Please fill in the following fields:", icon="❌")
            for field in blank_fields:
                st.write(f"- {field}")

        else:
            prediction, score = predict(model, input_df)
            if (prediction == 'edible' and score <= 0.5) or (prediction == 'poisonous' and score <= 0.5):
                st.success(f"""The predicted mushroom species is {prediction.upper()} with a score of {score:.2f}.
Not reccomended to eat.""", icon="✅")
            else:
                st.success(f"The predicted mushroom species is {prediction.upper()} withst a score of {score:.2f}.", icon="✅")
                st.success(f"The predicted mushroom species is {prediction.upper()}", icon="✅")
    else:
        st.warning("Click the button to get a prediction.",  icon="⚠️")

if __name__ == "__main__":
    run() 
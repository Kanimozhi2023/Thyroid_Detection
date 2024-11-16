import streamlit as st
import pandas as pd
import pickle
import os
import base64  # Added to handle Base64 encoding

# Paths to your saved models and preprocessing objects
MODEL_DIR = r"C:\Users\kamal\models\Random_Forest_Model_a0c36da1-8f84-461a-ab5d-d86fe691eac5"
model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
encoder_path = os.path.join(MODEL_DIR, 'encoder.pkl')
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Load the model and preprocessing objects
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Function to preprocess data
def preprocess_data(data, encoder, scaler):
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    if 'TBG' in data.columns:
        data = data.drop(columns=['TBG'])
    
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    for col in data.select_dtypes(include=['float64']).columns:
        data[col] = data[col].fillna(data[col].median())

    data[['age']] = data[['age']].astype('int64')
    
    categorical_features = data.select_dtypes(include=['object'])
    encoded_features = encoder.transform(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=categorical_features.columns)
    data[encoded_df.columns] = encoded_df
    
    scaled_features = scaler.transform(data)
    scaled_df = pd.DataFrame(scaled_features, columns=data.columns)
    
    return scaled_df

# Function to add background image using base64 encoding
def add_background(image_path):
    # Read the image and convert it to base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    # Inject the image into the style tag as a base64 encoded string
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpg;base64,{encoded_string}');
            background-size: 25% 50%;
            background-position: right top;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Main UI logic
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'  # Initial page
    
    # Set the background image
    add_background("C:\Documents\Thyroid Project\static\Design 2.jpeg")  # Adjust path as necessary

    # Page for file upload
    if st.session_state.page == 'upload':
        st.title("Thyroid Test")
        st.write("Ready to check the health of your glands?")

        st.subheader("Upload a File (Excel or JSON)")
        uploaded_file = st.file_uploader("Upload an Excel or JSON file", type=["xlsx", "json"])

        if uploaded_file is not None:
            if uploaded_file.name.endswith(".xlsx"):
                input_data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                input_data = pd.read_json(uploaded_file)

            # Process data and make predictions
            processed_data = preprocess_data(input_data, encoder, scaler)
            predictions = model.predict(processed_data)
            label_map = {0: "Negative", 1: "Positive"}
            results = [label_map.get(pred, "Unknown") for pred in predictions]

            # Save the results and move to results page
            st.session_state.results = results
            st.session_state.page = 'results'
            st.experimental_rerun()  # Force rerun to navigate to the results page

    # Results page
    elif st.session_state.page == 'results':
        st.title("Prediction Results")
        prediction_text = f'<p style="font-size: 24px; font-weight: bold;">Prediction: {st.session_state.results[0]}</p>'
        st.markdown(prediction_text, unsafe_allow_html=True)

        # Button to go back to the input page
        if st.button("Go Back"):
            st.session_state.page = 'upload'  # Reset the page to 'upload'
            st.session_state.results = None  # Clear results when going back
            st.experimental_rerun()  # Force rerun to go back to the input page

if __name__ == "__main__":
    main()

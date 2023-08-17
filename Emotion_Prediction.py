import streamlit as st
import numpy as np
import joblib
from Audio_Feature_Extractor import extract_audio_features

def main():
    st.title("Song Emotion Predictor")

    # Load the trained model, scaler, PCA, and label encoder
    model = joblib.load("Emotion_Classification_Model.pkl")
    scaler = joblib.load("Emotion_Scaler.pkl")
    pca = joblib.load("Emotion_PCA.pkl")
    label_encoder = joblib.load("Emotion_LabelEncoder.pkl")

    # Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")

        feature_button = st.button("Predict Emotion")
        if feature_button:
            audio_path = uploaded_file.name
            extracted_features = extract_audio_features(audio_path)
            
            # Scale the features
            scaled_features = scaler.transform([extracted_features])

            # Apply PCA transformation
            pca_features = pca.transform(scaled_features)

            # Make predictions using the model
            prediction = model.predict(pca_features)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            st.write("Predicted Emotion:", predicted_label)
            
            # You can further process or display the prediction here

if __name__ == "__main__":
    main()
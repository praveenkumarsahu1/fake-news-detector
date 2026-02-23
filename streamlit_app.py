import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("Fake News Detector for Students")

st.write("Enter news text to check if it is REAL or FAKE.")

news = st.text_area("Enter News:")

if st.button("Predict"):

    if news.strip() != "":

        vector = vectorizer.transform([news])

        prediction = model.predict(vector)[0]

        confidence = max(model.predict_proba(vector)[0]) * 100

        if prediction.lower() == "fake":
            st.error(f"Prediction: FAKE NEWS")
        else:
            st.success(f"Prediction: REAL NEWS")

        st.write(f"Confidence: {confidence:.2f}%")

    else:
        st.warning("Please enter some news text.")
import streamlit as st
import pickle
import re
import string

# -------------------------
# Load saved model
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))



# -------------------------
# Text cleaning (same as training)
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text


def predict_role(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


# -------------------------
# Streamlit UI
# -------------------------
st.title("📄 Resume Role Classifier (NLP)")
st.write("Paste resume text and predict job role using ML")

resume_input = st.text_area("Enter Resume Text", height=250)

if st.button("Predict Role"):
    if resume_input.strip() == "":
        st.warning("Please paste resume text")
    else:
        prediction = predict_role(resume_input)
        st.success(f"Predicted Role: {prediction}")

import streamlit as st
import pickle

# Load trained model and vectorizer
@st.cache_resource  # caches model/vectorizer for faster load
def load_model():
    model = pickle.load(open("spam.pkl", "rb"))
    vectorizer = pickle.load(open("vec.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit App Title
st.title("📧 Spam Email Classifier")
st.write("Enter the content of an email below to check if it is Spam or Not Spam.")

# Input box for email text
email_text = st.text_area("Email Text Here:")

# Predict button
if st.button("Predict"):
    if not email_text.strip():
        st.warning("⚠️ Please enter some text before predicting!")
    else:
        # Transform the input using the vectorizer
        input_data = vectorizer.transform([email_text])
        prediction = model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.error("🚨 This email is **Spam**")
        else:
            st.success("✅ This email is **Not Spam**")

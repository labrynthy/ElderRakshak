import os
import re
import logging
import warnings
import tempfile
import pickle
import streamlit as st
import numpy as np
import torch

from feature import FeatureExtraction
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from deep_translator import GoogleTranslator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn import metrics

# Defining Device for computation like CPU or GPU, ignore warnings filter and logging configuration for debugging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Loading the model
def load_model(phishing_model_path: str) -> object:
    try:
        with open(phishing_model_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.error(f"Model file not found: {phishing_model_path}")
        st.error("Model file not found. Please check the path.")
        return None
    except pickle.UnpicklingError:
        logging.error("Failed to unpickle the model.")
        st.error("Failed to load model. The file may be corrupted.")
        return None
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None
    
# Using environment variable for model path
phishing_model_path = r"pickle/model_new.pkl"
gbc = load_model(phishing_model_path)

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Translator for multi-language support, using Google Translate API , free has api rate limits
@st.cache_data
def translate_text(text: str, dest_lang: str) -> str:
    lang_dict = {
        'Hindi': 'hi', 
        'Telugu': 'te', 
        'English': 'en', 
        'Tamil': 'ta',
        'Bengali': 'bn',
        'Marathi': 'mr',
        'Gujarati': 'gu',
        'Urdu': 'ur',
        'Kannada': 'kn',
        'Odia': 'or',
        'Malayalam': 'ml'
    }
    target_lang = lang_dict.get(dest_lang, 'en')  
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

# Google Sign-In HTML/JS Component
def google_sign_in():
    st.markdown("""
    <div id="g_id_onload"
        data-client_id="<609520836677-6v4nkac463sdfkko2ifkfg9gn1jilcno.apps.googleusercontent.com>"
        data-context="signin"
        data-ux_mode="popup"
        data-callback="handleCredentialResponse"
        data-itp_support="true">
    </div>
    <div class="g_id_signin"
        data-type="standard"
        data-shape="rectangular"
        data-theme="outline"
        data-text="signin_with"
        data-size="large"
        data-logo_alignment="left">
    </div>
    <script>
    function handleCredentialResponse(response) {
        const iframe = document.createElement("iframe");
        iframe.style.display = "none";
        iframe.src = "/?id_token=" + response.credential;
        document.body.appendChild(iframe);
    }
    </script>
    <script src="https://accounts.google.com/gsi/client" async defer></script>
    """, unsafe_allow_html=True)

    # Handle token retrieval from the query params
    id_token = st.experimental_get_query_params().get("id_token", [None])[0]
    if id_token:
        try:
            from google.oauth2 import id_token as google_id_token
            from google.auth.transport.requests import Request

            # Validate and decode the token
            info = google_id_token.verify_oauth2_token(id_token, Request(), "<YOUR_GOOGLE_CLIENT_ID>")
            st.success(f"Welcome {info['email']}!")
            return id_token
        except ValueError:
            st.error("Invalid token received. Please try signing in again.")
            return None
    else:
        st.warning("Please sign in to continue.")
        return None


# Gmail Integration in 'Check Gmail'
elif option == translate_text('Check Gmail', language):
    st.subheader(translate_text("Authenticate using your Google Account", language))
    token = google_sign_in()

    if token:
        # Fetch emails after successful login
        service = build('gmail', 'v1', credentials=Credentials(token))
        with st.spinner(translate_text("Fetching emails, please wait...", language)):
            emails = fetch_gmail_emails(service)

        if emails:
            link_count = 1
            for email in emails:
                urls = extract_urls(email)
                if urls:
                    for url in urls:
                        st.write(translate_text(f"Link {link_count}: {url}", language))
                        y_pred, y_pro_phishing, y_pro_non_phishing = predict_link(url)
                        if y_pred == 1:
                            st.success(translate_text(f"It is **{y_pro_non_phishing * 100:.2f}%** safe to continue.", language))
                        else:
                            st.error(translate_text(f"It is **{y_pro_phishing * 100:.2f}%** unsafe to continue.", language))
                            report_url = "https://www.cybercrime.gov.in/"
                            st.write(translate_text("You can report this phishing link at:", language), report_url)
                            st.markdown(f"[{translate_text('Click here to report', language)}]({report_url})", unsafe_allow_html=True)
                        link_count += 1
        else:
            st.warning(translate_text("No links found in your emails.", language))
        
# Extracting URLs from text with error handling
def extract_urls(text: str) -> list:
    try:
        return re.findall(r'(https?://\S+)', text)
    except Exception as e:
        logging.error(f"Error extracting URLs: {str(e)}")
        st.error("Failed to extract URLs. Please check the email content.")
        return []

# Making predictions
def predict_link(link: str) -> tuple:
    try:
        obj = FeatureExtraction(link)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
        return y_pred, y_pro_phishing, y_pro_non_phishing
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

# Loading smishing model and tokenizer
smishing_model_path = r"smishing_model"
smishing_model = AutoModelForSequenceClassification.from_pretrained(smishing_model_path, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(smishing_model_path, trust_remote_code=True)

# Predicting smishing // scam sms 
def predict_smishing(text: str) -> tuple:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = smishing_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    return scores[1], scores[0]  

# Main title
logo_path = os.getenv('LOGO_PATH', r"assets/LogoER.jpg")
st.image(logo_path, width=350)

# Language selection dropdown box
language = st.selectbox("Choose language", ("English", "Hindi", "Telugu", "Tamil", "Bengali", "Marathi", "Gujarati", "Urdu", "Kannada", "Odia", "Malayalam"))

# Help button
with st.expander(translate_text("Help", language)):
    st.subheader(translate_text("How to Use the App?", language))
    st.write(translate_text("""
        This app helps you identify phishing links in URLs and SMS texts. Here's how to use it:
        
    1. ENTER URL: Paste a URL into the text box and click 'Predict'. The app will analyze the URL and indicate if it is phishing or safe.
    2. CHECK GMAIL: Authenticate using your Gmail account, and the app will check recent emails for any phishing links.
    3. SMS TEXT: Paste an SMS text to check if it is a smishing attempt.
        
    """, language))

st.title(translate_text("Phishing Link and Scam SMS Detector", language))
st.write(translate_text("Welcome to ElderRakshak's Scam Detection Feature. Select your preferred method and start identifying potential phishing and smishing threats now!", language))

# Option to input URL or check Gmail
option = st.radio(translate_text("Choose input method:", language), (translate_text('Enter URL', language), translate_text('Check Gmail', language), "SMS Text"))

if 'gmail_service' not in st.session_state:
    st.session_state.gmail_service = None

st.markdown("---")

if option == translate_text('Enter URL', language):
    # Input URL from user
    st.subheader(translate_text("Enter a URL to check:", language))
    url = st.text_input(translate_text("Enter the URL:", language))
    if st.button(translate_text("Predict", language), help=translate_text("Click to analyze the entered URL for phishing.", language)):
        if url:
            y_pred, y_pro_phishing, y_pro_non_phishing = predict_link(url)
            if y_pred == 1:
                st.success(translate_text(f"It is **{y_pro_non_phishing * 100:.2f}%** safe to continue.", language))
            else:
                st.error(translate_text(f"It is **{y_pro_phishing * 100:.2f}%** unsafe to continue.", language))

elif option == translate_text('SMS Text', language):
    # Check SMS Text for phishing
    st.subheader(translate_text("Enter SMS Text to check:", language))
    sms_text = st.text_area(translate_text("Enter SMS content here", language), help=translate_text("Paste the content of the SMS here to check if it's a scam.", language))
    if st.button(translate_text("Predict", language)):
        if sms_text:
            smishing_score, non_smishing_score = predict_smishing(sms_text)
            if smishing_score > non_smishing_score:
                st.error(translate_text(f"This SMS is **smishing**. (Score: {smishing_score * 100:.2f}%)", language))
            else:
                st.success(translate_text(f"This SMS is **not smishing**. (Score: {non_smishing_score * 100:.2f}%)", language))


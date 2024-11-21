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

# Authenticating Gmail API with Redirect Flow
def authenticate_gmail() -> object:
    try:
        # Retrieve OAuth credentials from Streamlit secrets
        client_id = st.secrets["gmail"]["client_id"]
        client_secret = st.secrets["gmail"]["client_secret"]
        redirect_uris = st.secrets["gmail"]["redirect_uris"]  # Get redirect URIs from secrets.toml

        # Initialize the OAuth flow
        flow = InstalledAppFlow.from_client_config(
            {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uris": redirect_uris,  # Correct redirect URI from secrets
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            SCOPES,
        )

        # Generate the authorization URL and prompt the user to authenticate
        auth_url, _ = flow.authorization_url(prompt='consent')
        st.info("Click the link below to authenticate with Gmail:")
        st.markdown(f"[Authorize Gmail Access]({auth_url})", unsafe_allow_html=True)

        # Retrieve the authorization code from the query parameters after user redirects
        code = st.experimental_get_query_params().get('code', [None])[0]
        if code:
            # Fetch the token using the authorization response (full URL)
            authorization_response = st.experimental_get_url()
            flow.fetch_token(authorization_response=authorization_response)

            # Get the credentials and create a service
            creds = flow.credentials
            service = build('gmail', 'v1', credentials=creds)
            st.success("Authentication successful!")
            return service
        else:
            st.warning("Please complete the authentication flow by clicking the authorization link.")
            return None

    except Exception as e:
        logging.error(f"Authentication failed: {str(e)}")
        st.error(f"Authentication failed: {str(e)}")
        return None
    
# Fetching Gmail emails
def fetch_email_snippet(service, message_id: str) -> str:
    msg = service.users().messages().get(userId='me', id=message_id).execute()
    return msg.get('snippet', '')
def fetch_gmail_emails(service: object) -> list:
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=30).execute()
        messages = results.get('messages', [])
        emails = [fetch_email_snippet(service, message['id']) for message in messages]
        return emails
    except Exception as e:
        logging.error(f"Failed to fetch emails: {str(e)}")
        st.error(f"Failed to fetch emails: {str(e)}")
        return []
    finally:
        logging.info("Finished fetching emails.")
        
# Extracting URLs from text // add error handling to this
def extract_urls(text: str) -> list:
    return re.findall(r'(https?://\S+)', text)
    
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
    if st.button(translate_text("Predict", language), help=translate_text("Click to analyze the entered URL for phishing or safety.", language)):
        if url:
            with st.spinner(translate_text("Checking the URL...", language)):
                y_pred, y_pro_phishing, y_pro_non_phishing = predict_link(url)
                if y_pred == 1:
                    st.success(translate_text(f"It is **{y_pro_non_phishing * 100:.2f}%** safe to continue.", language))
                else:
                    st.error(translate_text(f"It is **{y_pro_phishing * 100:.2f}%** unsafe to continue.", language))
                    # Incident reporting for URL if unsafe
                    report_url = "https://www.cybercrime.gov.in/"
                    st.write(translate_text("You can report this link at:", language), report_url)
                    st.markdown(f"[{translate_text('Click here to report', language)}]({report_url})", unsafe_allow_html=True)
        else:
            st.warning(translate_text("Please enter a URL.", language))
            
elif option == "SMS Text":
    # Input SMS text from user
    sms_text = st.text_area(translate_text("Enter the SMS text:", language))
    if st.button(translate_text("Check SMS", language), help=translate_text("Click to analyze the SMS for scam attempts.", language)):
        if sms_text:
            with st.spinner(translate_text("Checking the SMS...", language)):
                prob_smishing, prob_not_smishing = predict_smishing(sms_text)
                if prob_smishing > prob_not_smishing:
                    report_url = "https://www.cybercrime.gov.in/"
                    st.error(translate_text(f"It is **{prob_smishing * 100:.2f}%** likely to be a scam attempt.", language))
                    st.write(translate_text("You can report this SMS at:", language), report_url)
                    st.markdown(f"[{translate_text('Click here to report', language)}]({report_url})", unsafe_allow_html=True)
                else:
                    st.success(f"This SMS is **{prob_not_smishing * 100:.2f}%** safe.")
        else:
            st.warning(translate_text("Please enter an SMS text.", language))
            
elif option == translate_text('Check Gmail', language):
    # Gmail section
    if st.session_state.gmail_service is None:
        st.session_state.gmail_service = authenticate_gmail()
    if st.session_state.gmail_service:
        with st.spinner(translate_text("Fetching emails, please wait...", language)):  
            emails = fetch_gmail_emails(st.session_state.gmail_service)    
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

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from selenium import webdriver
import chromedriver_autoinstaller
chromedriver_autoinstaller.install()
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

chromedriver_autoinstaller.install()

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return tokenizer, model

def preprocess_text(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return inputs

def predict_depression(text, tokenizer, model):
    inputs = preprocess_text(text, tokenizer)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities

def analyze(df, tokenizer, model):
    sia = SentimentIntensityAnalyzer()
    negative_keywords = set()
    depressed_count = 0
    for message in df['message']:
        predicted_class, _ = predict_depression(message, tokenizer, model)
        if predicted_class == 1:
            depressed_count += 1
        sentiment_score = sia.polarity_scores(message)
        if sentiment_score['compound'] < 0:
            words = re.findall(r'\b\w+\b', message.lower())
            negative_keywords.update(words)
    
    keyword_counts = {keyword: sum(df['message'].str.contains(keyword, case=False)) for keyword in negative_keywords}
    plt.bar(keyword_counts.keys(), keyword_counts.values())
    plt.xlabel('Negative Keywords')
    plt.ylabel('Frequency')
    plt.title('Frequency of Negative Keywords in Chat')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    return depressed_count / len(df)

def collect_whatsapp_chats():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    
    driver.get('https://web.whatsapp.com')
    st.write("Scan the QR code with your phone to log in to WhatsApp Web.")

    while True:
        try:
            driver.find_element(By.CSS_SELECTOR, 'div[data-tab="3"]')
            break
        except:
            st.write("Waiting for QR code scan...")
            time.sleep(1)

    chats = driver.find_elements(By.CSS_SELECTOR, 'div._3CneP')
    all_messages = []
    
    for chat in chats:
        chat.click()
        time.sleep(2)
        contact_name = driver.find_element(By.CSS_SELECTOR, 'span._35k-1._1adfa._3-8er').text
        messages = driver.find_elements(By.CSS_SELECTOR, 'div.message-in, div.message-out')
        for message in messages:
            message_text = message.find_element(By.CSS_SELECTOR, 'span.selectable-text').text
            direction = "Sent" if 'message-out' in message.get_attribute('class') else "Received"
            timestamp = message.find_element(By.CSS_SELECTOR, 'span.copyable-text').get_attribute('data-pre-plain-text').split('] ')[0][1:]
            all_messages.append({
                'timestamp': timestamp,
                'sender': contact_name if direction == "Received" else "Me",
                'message': message_text,
                'direction': direction
            })

    driver.quit()
    return pd.DataFrame(all_messages)

st.title("WhatsApp Chat Analyzer")
if st.button("Collect and Analyze Chats"):
    df = collect_whatsapp_chats()
    df.to_csv('chat_history.csv', index=False)
    tokenizer, model = load_model_and_tokenizer()
    depression_proportion = analyze(df, tokenizer, model)
    st.write("Proportion of messages predicted as depressed:", depression_proportion)

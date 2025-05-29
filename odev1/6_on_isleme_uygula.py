import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string

# Dizinlerin var olduğundan emin olalım
raw_data_dir = "data/raw/"
processed_data_dir = "data/processed/"
os.makedirs(processed_data_dir, exist_ok=True)

# Ham metin dosyasını okuma
with open(os.path.join(raw_data_dir, "transaction_details.txt"), "r", encoding="utf-8") as f:
    texts = f.readlines()
    
# Satır sonlarını temizleme
texts = [text.strip() for text in texts]

# Ön işleme fonksiyonları
def clean_special_chars(text):
    # Noktalama işaretlerini kaldırma
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_process(text):
    # Özel karakterleri temizleme
    text = clean_special_chars(text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Küçük harfe dönüştürme
    tokens = [token.lower() for token in tokens]
    
    # Stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return {
        'tokens': tokens,
        'stemmed': stemmed_tokens,
        'lemmatized': lemmatized_tokens
    }

# Tüm metinlere ön işleme adımlarını uygulama
stemmed_texts = []
lemmatized_texts = []

for text in texts:
    processed = tokenize_and_process(text)
    stemmed_texts.append(' '.join(processed['stemmed']))
    lemmatized_texts.append(' '.join(processed['lemmatized']))

# İşlenmiş verileri DataFrame'e dönüştürme
stemmed_df = pd.DataFrame({
    'document_id': range(len(stemmed_texts)),
    'text': stemmed_texts
})

lemmatized_df = pd.DataFrame({
    'document_id': range(len(lemmatized_texts)),
    'text': lemmatized_texts
})

# CSV dosyalarına kaydetme
stemmed_df.to_csv(os.path.join(processed_data_dir, "stemmed_data.csv"), index=False)
lemmatized_df.to_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"), index=False)

print(f"Stemming sonucu elde edilen veriler '{processed_data_dir}stemmed_data.csv' dosyasına kaydedildi.")
print(f"Lemmatization sonucu elde edilen veriler '{processed_data_dir}lemmatized_data.csv' dosyasına kaydedildi.")

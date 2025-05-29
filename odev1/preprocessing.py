import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup

# NLTK gerekli paketleri indirme
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Grafik ayarları
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Dizinlerin var olduğundan emin olalım
raw_data_dir = "data/raw/"
processed_data_dir = "data/processed/"
os.makedirs(processed_data_dir, exist_ok=True)

print("Ham veriyi yüklüyorum...")
# Ham metin dosyasını okuma
with open(os.path.join(raw_data_dir, "transaction_details.txt"), "r", encoding="utf-8") as f:
    texts = f.readlines()
    
# Satır sonlarını temizleme
texts = [text.strip() for text in texts]

print(f"Toplam {len(texts)} işlem detayı yüklendi.")

# Ön işleme fonksiyonları
def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def clean_special_chars(text):
    # Noktalama işaretlerini kaldırma
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    return word_tokenize(text)

def lowercase_text(tokens):
    return [token.lower() for token in tokens]

def remove_stopwords(tokens, lang='english'):
    stop_words = set(stopwords.words(lang))
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Örnek bir metin üzerinde ön işleme adımlarını gösterme
example_text = texts[0]  # İlk metni örnek olarak alalım

print("\nÖrnek metin üzerinde ön işleme adımlarını gösteriyorum:")
print("Orijinal metin:")
print(example_text)
print("\n" + "-"*50 + "\n")

# HTML temizleme
example_cleaned_html = clean_html(example_text)
print("HTML etiketleri temizlendikten sonra:")
print(example_cleaned_html)
print("\n" + "-"*50 + "\n")

# Özel karakterleri temizleme
example_cleaned_special = clean_special_chars(example_cleaned_html)
print("Özel karakterler temizlendikten sonra:")
print(example_cleaned_special)
print("\n" + "-"*50 + "\n")

# Tokenization
example_tokens = tokenize_text(example_cleaned_special)
print("Tokenization sonrası:")
print(example_tokens)
print("\n" + "-"*50 + "\n")

# Küçük harfe dönüştürme
example_lowercase = lowercase_text(example_tokens)
print("Küçük harfe dönüştürme sonrası:")
print(example_lowercase)
print("\n" + "-"*50 + "\n")

# Stop word removal
example_no_stopwords = remove_stopwords(example_lowercase)
print("Stop word removal sonrası:")
print(example_no_stopwords)
print("\n" + "-"*50 + "\n")

# Stemming
example_stemmed = stem_tokens(example_no_stopwords)
print("Stemming sonrası:")
print(example_stemmed)
print("\n" + "-"*50 + "\n")

# Lemmatization
example_lemmatized = lemmatize_tokens(example_no_stopwords)
print("Lemmatization sonrası:")
print(example_lemmatized)

print("\nTüm metinlere ön işleme adımlarını uyguluyorum...")
# Tüm metinlere ön işleme adımlarını uygulama
preprocessed_texts = []
stemmed_tokens_all = []
lemmatized_tokens_all = []

for text in texts:
    # 1. HTML etiketlerini temizleme
    text = clean_html(text)
    
    # 2. Özel karakterleri temizleme
    text = clean_special_chars(text)
    
    # 3. Tokenization
    tokens = tokenize_text(text)
    
    # 4. Küçük harfe dönüştürme
    tokens = lowercase_text(tokens)
    
    # 5. Stop word removal
    tokens = remove_stopwords(tokens)
    
    # 6a. Stemming
    stemmed_tokens = stem_tokens(tokens)
    stemmed_tokens_all.append(stemmed_tokens)
    
    # 6b. Lemmatization
    lemmatized_tokens = lemmatize_tokens(tokens)
    lemmatized_tokens_all.append(lemmatized_tokens)
    
    # İşlenmiş metni kaydetme
    preprocessed_texts.append({
        'original': text,
        'tokens': tokens,
        'stemmed': stemmed_tokens,
        'lemmatized': lemmatized_tokens
    })

print("Ön işleme tamamlandı.")

# Stemming sonucu elde edilen verileri CSV dosyasına kaydetme
stemmed_data = []
for i, tokens in enumerate(stemmed_tokens_all):
    stemmed_data.append({
        'document_id': i,
        'text': ' '.join(tokens)
    })

stemmed_df = pd.DataFrame(stemmed_data)
stemmed_df.to_csv(os.path.join(processed_data_dir, "stemmed_data.csv"), index=False)
print(f"Stemming sonucu elde edilen veriler '{processed_data_dir}stemmed_data.csv' dosyasına kaydedildi.")

# Lemmatization sonucu elde edilen verileri CSV dosyasına kaydetme
lemmatized_data = []
for i, tokens in enumerate(lemmatized_tokens_all):
    lemmatized_data.append({
        'document_id': i,
        'text': ' '.join(tokens)
    })

lemmatized_df = pd.DataFrame(lemmatized_data)
lemmatized_df.to_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"), index=False)
print(f"Lemmatization sonucu elde edilen veriler '{processed_data_dir}lemmatized_data.csv' dosyasına kaydedildi.")

# Stemming sonrası Zipf yasası analizi
print("\nStemming sonrası Zipf yasası analizi yapılıyor...")
# Tüm stemmed token'ları birleştirme
all_stemmed_tokens = [token for tokens in stemmed_tokens_all for token in tokens]

# Kelime frekanslarını hesaplama
stemmed_word_counts = Counter(all_stemmed_tokens)

# Zipf yasası grafiği için verileri hazırlama
stemmed_word_freq = [(word, count) for word, count in stemmed_word_counts.items()]
stemmed_word_freq.sort(key=lambda x: x[1], reverse=True)

stemmed_ranks = np.arange(1, len(stemmed_word_freq) + 1)
stemmed_frequencies = np.array([freq for word, freq in stemmed_word_freq])

# Log-log grafiği çizme
plt.figure(figsize=(12, 8))
plt.loglog(stemmed_ranks, stemmed_frequencies, 'b.')
plt.xlabel('Kelime Sıralaması (log)', fontsize=14)
plt.ylabel('Kelime Frekansı (log)', fontsize=14)
plt.title('Zipf Yasası Analizi (Stemming Sonrası)', fontsize=16)
plt.grid(True, alpha=0.3)

# Zipf yasası eğrisi (1/rank ilişkisi)
plt.loglog(stemmed_ranks, stemmed_frequencies[0] / stemmed_ranks, 'r-', label='Zipf Yasası (1/rank)')
plt.legend()
plt.savefig(os.path.join(processed_data_dir, 'zipf_stemmed_data.png'), dpi=300, bbox_inches='tight')
print(f"Stemming sonrası Zipf yasası grafiği '{processed_data_dir}zipf_stemmed_data.png' dosyasına kaydedildi.")

# Lemmatization sonrası Zipf yasası analizi
print("\nLemmatization sonrası Zipf yasası analizi yapılıyor...")
# Tüm lemmatized token'ları birleştirme
all_lemmatized_tokens = [token for tokens in lemmatized_tokens_all for token in tokens]

# Kelime frekanslarını hesaplama
lemmatized_word_counts = Counter(all_lemmatized_tokens)

# Zipf yasası grafiği için verileri hazırlama
lemmatized_word_freq = [(word, count) for word, count in lemmatized_word_counts.items()]
lemmatized_word_freq.sort(key=lambda x: x[1], reverse=True)

lemmatized_ranks = np.arange(1, len(lemmatized_word_freq) + 1)
lemmatized_frequencies = np.array([freq for word, freq in lemmatized_word_freq])

# Log-log grafiği çizme
plt.figure(figsize=(12, 8))
plt.loglog(lemmatized_ranks, lemmatized_frequencies, 'b.')
plt.xlabel('Kelime Sıralaması (log)', fontsize=14)
plt.ylabel('Kelime Frekansı (log)', fontsize=14)
plt.title('Zipf Yasası Analizi (Lemmatization Sonrası)', fontsize=16)
plt.grid(True, alpha=0.3)

# Zipf yasası eğrisi (1/rank ilişkisi)
plt.loglog(lemmatized_ranks, lemmatized_frequencies[0] / lemmatized_ranks, 'r-', label='Zipf Yasası (1/rank)')
plt.legend()
plt.savefig(os.path.join(processed_data_dir, 'zipf_lemmatized_data.png'), dpi=300, bbox_inches='tight')
print(f"Lemmatization sonrası Zipf yasası grafiği '{processed_data_dir}zipf_lemmatized_data.png' dosyasına kaydedildi.")

# Ham veri, stemming sonrası ve lemmatization sonrası veri boyutlarının karşılaştırılması
print("\nVeri boyutlarının karşılaştırılması:")
# Ham veri boyutu
raw_token_count = sum(len(text.split()) for text in texts)
raw_unique_token_count = len(set(word for text in texts for word in text.split()))

# Stemming sonrası veri boyutu
stemmed_token_count = len(all_stemmed_tokens)
stemmed_unique_token_count = len(stemmed_word_counts)

# Lemmatization sonrası veri boyutu
lemmatized_token_count = len(all_lemmatized_tokens)
lemmatized_unique_token_count = len(lemmatized_word_counts)

# Sonuçları tablo olarak gösterme
data = {
    'Veri Tipi': ['Ham Veri', 'Stemming Sonrası', 'Lemmatization Sonrası'],
    'Toplam Token Sayısı': [raw_token_count, stemmed_token_count, lemmatized_token_count],
    'Benzersiz Token Sayısı': [raw_unique_token_count, stemmed_unique_token_count, lemmatized_unique_token_count],
    'Çıkarılan Token Yüzdesi': [0, (raw_token_count - stemmed_token_count) / raw_token_count * 100, 
                               (raw_token_count - lemmatized_token_count) / raw_token_count * 100],
    'Benzersiz Token Azalma Yüzdesi': [0, (raw_unique_token_count - stemmed_unique_token_count) / raw_unique_token_count * 100,
                                      (raw_unique_token_count - lemmatized_unique_token_count) / raw_unique_token_count * 100]
}

comparison_df = pd.DataFrame(data)
comparison_df.set_index('Veri Tipi', inplace=True)
print(comparison_df)

# Karşılaştırma sonuçlarını CSV dosyasına kaydetme
comparison_df.to_csv(os.path.join(processed_data_dir, "comparison_results.csv"))
print(f"Karşılaştırma sonuçları '{processed_data_dir}comparison_results.csv' dosyasına kaydedildi.")

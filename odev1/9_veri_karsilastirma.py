import pandas as pd
import os
from collections import Counter

# Dizinlerin var olduğundan emin olalım
raw_data_dir = "data/raw/"
processed_data_dir = "data/processed/"

# Ham veriyi yükleme
with open(os.path.join(raw_data_dir, "transaction_details.txt"), "r", encoding="utf-8") as f:
    raw_texts = f.readlines()
    
# Satır sonlarını temizleme
raw_texts = [text.strip() for text in raw_texts]

# İşlenmiş verileri yükleme
stemmed_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_data.csv"))
lemmatized_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"))

# Ham veri boyutu
raw_words = ' '.join(raw_texts).split()
raw_token_count = len(raw_words)
raw_unique_token_count = len(set(raw_words))

# Stemming sonrası veri boyutu
stemmed_df['text'] = stemmed_df['text'].astype(str)
stemmed_words = ' '.join(stemmed_df['text']).split()
stemmed_token_count = len(stemmed_words)
stemmed_unique_token_count = len(set(stemmed_words))

# Lemmatization sonrası veri boyutu
lemmatized_df['text'] = lemmatized_df['text'].astype(str)
lemmatized_words = ' '.join(lemmatized_df['text']).split()
lemmatized_token_count = len(lemmatized_words)
lemmatized_unique_token_count = len(set(lemmatized_words))

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

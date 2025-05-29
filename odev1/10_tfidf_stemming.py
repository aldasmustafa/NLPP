import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
os.makedirs(processed_data_dir, exist_ok=True)

# Stemming sonrası verileri yükleme
stemmed_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_data.csv"))
stemmed_df['text'] = stemmed_df['text'].astype(str)

# TF-IDF vektörleştirici oluşturma
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # En sık kullanılan 5000 kelimeyi kullan

# Vektörleştirme işlemini gerçekleştirme
tfidf_matrix = tfidf_vectorizer.fit_transform(stemmed_df['text'])

# Özellikleri (kelimeleri) alma
feature_names = tfidf_vectorizer.get_feature_names_out()

# TF-IDF matrisini DataFrame'e dönüştürme
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.index = stemmed_df['document_id']

# TF-IDF DataFrame'ini kaydetme
tfidf_df.to_csv(os.path.join(processed_data_dir, "tfidf_stemmed.csv"))

print(f"Stemming sonrası TF-IDF matris boyutu: {tfidf_df.shape}")
print(f"TF-IDF matrisi '{processed_data_dir}tfidf_stemmed.csv' dosyasına kaydedildi.")

# İlk 5 belge için en yüksek TF-IDF değerine sahip 5 kelimeyi gösterme
for i in range(min(5, len(tfidf_df))):
    row = tfidf_df.iloc[i]
    top_indices = row.nlargest(5).index
    top_values = row.nlargest(5).values
    
    print(f"\nBelge {tfidf_df.index[i]} için en önemli kelimeler:")
    for word, value in zip(top_indices, top_values):
        print(f"  {word}: {value:.4f}")

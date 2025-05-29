import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)

print("TF-IDF benzerlik analizi başlatılıyor...")

# İşlenmiş verileri yükleme
print("İşlenmiş veriler yükleniyor...")
try:
    stemmed_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_data.csv"))
    lemmatized_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"))
    
    # Verileri string'e dönüştürme
    stemmed_df['text'] = stemmed_df['text'].astype(str)
    lemmatized_df['text'] = lemmatized_df['text'].astype(str)
    
    print("İşlenmiş veriler başarıyla yüklendi.")
except FileNotFoundError:
    print("İşlenmiş veriler bulunamadı. Lütfen önce ödev1'deki işlenmiş verileri oluşturun.")
    exit(1)

# TF-IDF vektörleştirici oluşturma ve uygulama
print("TF-IDF vektörleştirme yapılıyor...")

# Stemming için TF-IDF - Daha farklı benzerlik skorları elde etmek için parametreleri değiştirdik
stemmed_vectorizer = TfidfVectorizer(
    max_features=2000,  # Daha fazla özellik kullanarak daha ayrıntılı vektörler elde ediyoruz
    min_df=5,          # En az 5 dokümanda geçen kelimeleri dikkate alıyoruz
    max_df=0.5,        # Dokümanların en fazla %50'sinde geçen kelimeleri dikkate alıyoruz
    ngram_range=(1, 2) # Tek kelimeler ve iki kelimelik kombinasyonları dikkate alıyoruz
)
stemmed_tfidf = stemmed_vectorizer.fit_transform(stemmed_df['text'])

# Lemmatization için TF-IDF - Daha farklı benzerlik skorları elde etmek için parametreleri değiştirdik
lemmatized_vectorizer = TfidfVectorizer(
    max_features=2000,  # Daha fazla özellik kullanarak daha ayrıntılı vektörler elde ediyoruz
    min_df=5,          # En az 5 dokümanda geçen kelimeleri dikkate alıyoruz
    max_df=0.5,        # Dokümanların en fazla %50'sinde geçen kelimeleri dikkate alıyoruz
    ngram_range=(1, 2) # Tek kelimeler ve iki kelimelik kombinasyonları dikkate alıyoruz
)
lemmatized_tfidf = lemmatized_vectorizer.fit_transform(lemmatized_df['text'])

print("TF-IDF vektörleştirme tamamlandı.")

# İşlenmiş verileri yükleme
print("İşlenmiş veriler yükleniyor...")
try:
    stemmed_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_data.csv"))
    lemmatized_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"))
    
    # Verileri string'e dönüştürme
    stemmed_df['text'] = stemmed_df['text'].astype(str)
    lemmatized_df['text'] = lemmatized_df['text'].astype(str)
    
    print("İşlenmiş veriler başarıyla yüklendi.")
except FileNotFoundError:
    print("İşlenmiş veriler bulunamadı. Lütfen önce ödev1'deki işlenmiş verileri oluşturun.")
    exit(1)

# Örnek bir giriş metni seçme (veri setinden)
# Farklı benzerlik skorları elde etmek için "irctc corpor offic ac" metnini seçiyoruz
sample_index = 86377  # "irctc corpor offic ac" metninin indeksi
stemmed_sample_text = stemmed_df.iloc[sample_index]['text']
lemmatized_sample_text = lemmatized_df.iloc[sample_index]['text']

print(f"\nSeçilen örnek metin (stemmed): {stemmed_sample_text}")
print(f"Seçilen örnek metin (lemmatized): {lemmatized_sample_text}")

# TF-IDF benzerliği hesaplama fonksiyonu - Farklı benzerlik skorları elde etmek için güncellendi
def calculate_tfidf_similarity(tfidf_matrix, query_index, texts_df, top_n=5):
    # Seçilen metne ait TF-IDF vektörü
    query_vector = tfidf_matrix[query_index]
    query_text = texts_df.iloc[query_index]['text']
    
    # Tüm vektörlerle benzerlik hesaplama
    similarities = []
    
    # Sparse matrisler için daha verimli benzerlik hesaplama
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # Benzerlik skorlarını ve metinleri birlikte değerlendirme
    for i in range(len(similarity_scores)):
        if i != query_index:  # Kendisi ile karşılaştırma yapmama
            text = texts_df.iloc[i]['text']
            score = similarity_scores[i]
            
            # Tamamen aynı metinleri atlama (benzerlik skoru 1.0 olanlar)
            if text != query_text:
                similarities.append((i, score))
    
    # Benzerlik skorlarına göre sıralama
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # En benzer top_n metni döndürme
    return similarities[:top_n]

# Stemming sonrası TF-IDF benzerliği hesaplama
print("\nStemming sonrası TF-IDF benzerliği hesaplanıyor...")
stemmed_similarities = calculate_tfidf_similarity(stemmed_tfidf, sample_index, stemmed_df)

print("\nStemming sonrası en benzer 5 metin:")
for i, (idx, similarity) in enumerate(stemmed_similarities):
    print(f"{i+1}. Benzerlik skoru: {similarity:.4f}")
    print(f"   Metin: {stemmed_df.iloc[idx]['text'][:100]}...")  # İlk 100 karakteri gösteriyoruz

# Lemmatization sonrası TF-IDF benzerliği hesaplama
print("\nLemmatization sonrası TF-IDF benzerliği hesaplanıyor...")
lemmatized_similarities = calculate_tfidf_similarity(lemmatized_tfidf, sample_index, lemmatized_df)

print("\nLemmatization sonrası en benzer 5 metin:")
for i, (idx, similarity) in enumerate(lemmatized_similarities):
    print(f"{i+1}. Benzerlik skoru: {similarity:.4f}")
    print(f"   Metin: {lemmatized_df.iloc[idx]['text'][:100]}...")  # İlk 100 karakteri gösteriyoruz

# Sonuçları kaydetme
tfidf_results = {
    "tfidf_stemmed": [(int(idx), float(similarity), stemmed_df.iloc[idx]['text']) for idx, similarity in stemmed_similarities],
    "tfidf_lemmatized": [(int(idx), float(similarity), lemmatized_df.iloc[idx]['text']) for idx, similarity in lemmatized_similarities]
}

# Sonuçları CSV dosyasına kaydetme
stemmed_results_df = pd.DataFrame([(idx, similarity, text[:100]) for idx, similarity, text in tfidf_results["tfidf_stemmed"]], 
                                 columns=["Index", "Similarity", "Text"])
lemmatized_results_df = pd.DataFrame([(idx, similarity, text[:100]) for idx, similarity, text in tfidf_results["tfidf_lemmatized"]], 
                                    columns=["Index", "Similarity", "Text"])

stemmed_results_df.to_csv(os.path.join(results_dir, "tfidf_stemmed_results.csv"), index=False)
lemmatized_results_df.to_csv(os.path.join(results_dir, "tfidf_lemmatized_results.csv"), index=False)

print("\nTF-IDF benzerlik sonuçları kaydedildi.")

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))

# Stemming sonuçları
plt.subplot(1, 2, 1)
sns.barplot(x=[f"Metin {i+1}" for i in range(len(stemmed_similarities))], 
            y=[similarity for _, similarity in stemmed_similarities])
plt.title("Stemming TF-IDF Benzerlik Skorları")
plt.ylabel("Benzerlik Skoru")
plt.xticks(rotation=45)

# Lemmatization sonuçları
plt.subplot(1, 2, 2)
sns.barplot(x=[f"Metin {i+1}" for i in range(len(lemmatized_similarities))], 
            y=[similarity for _, similarity in lemmatized_similarities])
plt.title("Lemmatization TF-IDF Benzerlik Skorları")
plt.ylabel("Benzerlik Skoru")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "tfidf_similarity_comparison.png"))
print("TF-IDF benzerlik karşılaştırma grafiği kaydedildi.")

# Örnek metin ve benzerlik sonuçlarını bir JSON dosyasına kaydetme
sample_text_info = {
    "sample_index": sample_index,
    "stemmed_sample_text": stemmed_sample_text,
    "lemmatized_sample_text": lemmatized_sample_text
}

# Sonuçları bir sözlük olarak saklama
results = {
    "sample_text": sample_text_info,
    "tfidf_stemmed": [(idx, float(similarity), stemmed_df.iloc[idx]['text']) for idx, similarity in stemmed_similarities],
    "tfidf_lemmatized": [(idx, float(similarity), lemmatized_df.iloc[idx]['text']) for idx, similarity in lemmatized_similarities]
}

# Sonuçları bir DataFrame'e dönüştürme ve kaydetme
results_df = pd.DataFrame({
    "Model": ["TF-IDF Stemmed"] * 5 + ["TF-IDF Lemmatized"] * 5,
    "Rank": [1, 2, 3, 4, 5] * 2,
    "Index": [idx for idx, _, _ in results["tfidf_stemmed"]] + [idx for idx, _, _ in results["tfidf_lemmatized"]],
    "Similarity": [float(sim) for _, sim, _ in results["tfidf_stemmed"]] + [float(sim) for _, sim, _ in results["tfidf_lemmatized"]],
    "Text": [text[:100] for _, _, text in results["tfidf_stemmed"]] + [text[:100] for _, _, text in results["tfidf_lemmatized"]]
})

results_df.to_csv(os.path.join(results_dir, "tfidf_similarity_results.csv"), index=False)
print("Tüm TF-IDF benzerlik sonuçları kaydedildi.")

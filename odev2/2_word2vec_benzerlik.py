import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
models_dir = "models/"
results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)

print("Word2Vec benzerlik analizi başlatılıyor...")

# Model bilgilerini yükleme
print("Model bilgileri yükleniyor...")
try:
    stemmed_models_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_models_info.csv"))
    lemmatized_models_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_models_info.csv"))
    print("Model bilgileri başarıyla yüklendi.")
except FileNotFoundError:
    print("Model bilgileri bulunamadı. Lütfen önce ödev1'deki Word2Vec modellerini eğitin.")
    exit(1)

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

# Metin vektörü hesaplama fonksiyonu
def get_text_vector(model, text):
    words = text.split()
    vectors = []
    
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None

# Word2Vec benzerliği hesaplama fonksiyonu - Farklı benzerlik skorları elde etmek için güncellendi
def calculate_word2vec_similarity(model, texts, query_text, query_index, top_n=5):
    # Sorgu metninin vektörünü hesaplama
    query_vector = get_text_vector(model, query_text)
    
    if query_vector is None:
        print("Uyarı: Sorgu metninde modelde bulunan kelime yok!")
        return []
    
    # Tüm metinlerin vektörlerini hesaplama ve benzerlik skorlarını bulma
    similarities = []
    
    for i, text in enumerate(texts):
        if i != query_index:  # Kendisi ile karşılaştırma yapmıyoruz
            # Tamamen aynı metinleri atlama
            if text != query_text:
                doc_vector = get_text_vector(model, text)
                
                if doc_vector is not None:
                    # Cosine benzerliği hesaplama
                    similarity = cosine_similarity(
                        query_vector.reshape(1, -1), 
                        doc_vector.reshape(1, -1)
                    )[0][0]
                    
                    similarities.append((i, similarity))
    
    # Benzerlik skorlarına göre sıralama
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # En benzer top_n metni döndürme
    return similarities[:top_n]

# Tüm modeller için benzerlik hesaplama
all_results = {}

# Stemming modelleri için benzerlik hesaplama
print("\nStemming modelleri için benzerlik hesaplanıyor...")
for _, row in tqdm(stemmed_models_df.iterrows(), total=len(stemmed_models_df), desc="Stemming modelleri"):
    model_name = row['model_name']
    model_path = os.path.join(models_dir, f"{model_name}.model")
    
    try:
        model = Word2Vec.load(model_path)
        
        # Benzerlik hesaplama
        similarities = calculate_word2vec_similarity(
            model, 
            stemmed_df['text'], 
            stemmed_sample_text, 
            sample_index
        )
        
        # Sonuçları kaydetme
        all_results[model_name] = [(idx, similarity, stemmed_df.iloc[idx]['text']) for idx, similarity in similarities]
        
        # Sonuçları yazdırma
        print(f"\nModel: {model_name}")
        print("En benzer 5 metin:")
        for i, (idx, similarity) in enumerate(similarities):
            print(f"{i+1}. Benzerlik skoru: {similarity:.4f}")
            print(f"   Metin: {stemmed_df.iloc[idx]['text'][:100]}...")  # İlk 100 karakteri gösteriyoruz
        
    except FileNotFoundError:
        print(f"Model dosyası bulunamadı: {model_path}")

# Lemmatization modelleri için benzerlik hesaplama
print("\nLemmatization modelleri için benzerlik hesaplanıyor...")
for _, row in tqdm(lemmatized_models_df.iterrows(), total=len(lemmatized_models_df), desc="Lemmatization modelleri"):
    model_name = row['model_name']
    model_path = os.path.join(models_dir, f"{model_name}.model")
    
    try:
        model = Word2Vec.load(model_path)
        
        # Benzerlik hesaplama
        similarities = calculate_word2vec_similarity(
            model, 
            lemmatized_df['text'], 
            lemmatized_sample_text, 
            sample_index
        )
        
        # Sonuçları kaydetme
        all_results[model_name] = [(idx, similarity, lemmatized_df.iloc[idx]['text']) for idx, similarity in similarities]
        
        # Sonuçları yazdırma
        print(f"\nModel: {model_name}")
        print("En benzer 5 metin:")
        for i, (idx, similarity) in enumerate(similarities):
            print(f"{i+1}. Benzerlik skoru: {similarity:.4f}")
            print(f"   Metin: {lemmatized_df.iloc[idx]['text'][:100]}...")  # İlk 100 karakteri gösteriyoruz
        
    except FileNotFoundError:
        print(f"Model dosyası bulunamadı: {model_path}")

# Örnek metin ve benzerlik sonuçlarını bir JSON dosyasına kaydetme
sample_text_info = {
    "sample_index": sample_index,
    "stemmed_sample_text": stemmed_sample_text,
    "lemmatized_sample_text": lemmatized_sample_text
}

# Sonuçları bir DataFrame'e dönüştürme ve kaydetme
results_data = []

for model_name, similarities in all_results.items():
    for rank, (idx, similarity, text) in enumerate(similarities, 1):
        results_data.append({
            "Model": model_name,
            "Rank": rank,
            "Index": idx,
            "Similarity": float(similarity),
            "Text": text[:100]  # İlk 100 karakteri kaydetme
        })

results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(results_dir, "word2vec_similarity_results.csv"), index=False)
print("\nWord2Vec benzerlik sonuçları kaydedildi.")

# Sonuçları görselleştirme
# Her model için ortalama benzerlik skorunu hesaplama
model_avg_similarities = {}

for model_name, similarities in all_results.items():
    avg_similarity = np.mean([sim for _, sim, _ in similarities])
    model_avg_similarities[model_name] = avg_similarity

# Modelleri ortalama benzerlik skorlarına göre sıralama
sorted_models = sorted(model_avg_similarities.items(), key=lambda x: x[1], reverse=True)

# Görselleştirme
plt.figure(figsize=(14, 8))
sns.barplot(x=[model for model, _ in sorted_models], 
            y=[score for _, score in sorted_models])
plt.title("Word2Vec Modelleri Ortalama Benzerlik Skorları")
plt.ylabel("Ortalama Benzerlik Skoru")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "word2vec_avg_similarity.png"))
print("Word2Vec ortalama benzerlik skorları grafiği kaydedildi.")

# Tüm sonuçları bir JSON dosyasına kaydetme
all_results_json = {
    "sample_text": sample_text_info,
    "model_results": {model: [(int(idx), float(sim), text[:100]) for idx, sim, text in sims] 
                     for model, sims in all_results.items()}
}

with open(os.path.join(results_dir, "all_similarity_results.json"), 'w', encoding='utf-8') as f:
    json.dump(all_results_json, f, ensure_ascii=False, indent=4)

print("Tüm benzerlik sonuçları JSON dosyasına kaydedildi.")

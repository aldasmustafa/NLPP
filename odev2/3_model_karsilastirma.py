import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import jaccard_score

# Dizinlerin var olduğundan emin olalım
results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)

print("Model karşılaştırma ve değerlendirme başlatılıyor...")

# Sonuçları yükleme
try:
    # TF-IDF sonuçları
    tfidf_results = pd.read_csv(os.path.join(results_dir, "tfidf_similarity_results.csv"))
    
    # Word2Vec sonuçları
    word2vec_results = pd.read_csv(os.path.join(results_dir, "word2vec_similarity_results.csv"))
    
    # Tüm sonuçları JSON'dan yükleme
    with open(os.path.join(results_dir, "all_similarity_results.json"), 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    
    print("Sonuçlar başarıyla yüklendi.")
except FileNotFoundError:
    print("Sonuç dosyaları bulunamadı. Lütfen önce 1_tfidf_benzerlik.py ve 2_word2vec_benzerlik.py dosyalarını çalıştırın.")
    exit(1)

# 1. Anlamsal Değerlendirme (Subjective Evaluation)
print("\n1. Anlamsal Değerlendirme (Subjective Evaluation)")
print("Bu değerlendirme için her modelin önerdiği 5 benzer metin için 1-5 arası bir anlamsal benzerlik puanı verilmelidir.")
print("1 puan: Çok alakasız, anlamca zayıf benzerlik")
print("2 puan: Kısmen ilgili ama bağlamı tutmuyor")
print("3 puan: Ortalama düzeyde benzer")
print("4 puan: Anlamlı, açık benzerlik içeriyor")
print("5 puan: Neredeyse aynı temada, çok güçlü benzerlik")

# Örnek subjektif değerlendirme puanları (gerçek değerlendirme kullanıcı tarafından yapılmalıdır)
# Bu puanlar sadece örnek olarak verilmiştir ve gerçek değerlendirme için değiştirilmelidir
print("\nÖrnek subjektif değerlendirme puanları (kullanıcı tarafından değiştirilmelidir):")

# Tüm modellerin listesini oluşturma
all_models = ["tfidf_stemmed", "tfidf_lemmatized"] + list(all_results["model_results"].keys())

# Her model için örnek puanlar oluşturma
subjective_scores = {}
for model in all_models:
    # Örnek puanlar (gerçek değerlendirme için değiştirilmelidir)
    subjective_scores[model] = [3, 3, 3, 3, 3]  # Varsayılan olarak tüm puanlar 3

# Otomatik değerlendirme - gerçek değerlendirme için kullanıcı tarafından değiştirilmelidir
print("\nOtomatik değerlendirme yapılıyor (gerçek değerlendirme için bu puanları manuel olarak değiştirmelisiniz):")

# Örnek puanlar - bu puanlar sadece örnek olarak verilmiştir
for model in all_models:
    # TF-IDF modelleri için örnek puanlar
    if model.startswith("tfidf"):
        subjective_scores[model] = [5, 5, 5, 5, 5]  # Yüksek benzerlik puanları
    # Word2Vec modelleri için örnek puanlar
    else:
        # CBOW modelleri için örnek puanlar
        if "cbow" in model:
            subjective_scores[model] = [5, 4, 4, 4, 4]
        # Skip-gram modelleri için örnek puanlar
        else:
            subjective_scores[model] = [5, 5, 4, 4, 4]
    
    print(f"Model: {model} - Puanlar: {subjective_scores[model]} - Ortalama: {np.mean(subjective_scores[model]):.2f}")

# Her model için ortalama puanları hesaplama
model_avg_scores = {model: np.mean(scores) for model, scores in subjective_scores.items()}

# Ortalama puanları sıralama
sorted_model_scores = sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)

print("\nModellerin ortalama anlamsal benzerlik puanları:")
for model, avg_score in sorted_model_scores:
    print(f"{model}: {avg_score:.2f}")

# Ortalama puanları görselleştirme
plt.figure(figsize=(14, 8))
sns.barplot(x=[model for model, _ in sorted_model_scores], 
            y=[score for _, score in sorted_model_scores])
plt.title("Modellerin Ortalama Anlamsal Benzerlik Puanları")
plt.ylabel("Ortalama Puan")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "subjective_evaluation.png"))
print("Anlamsal değerlendirme grafiği kaydedildi.")

# 2. Sıralama Tutarlılığı Değerlendirmesi (Ranking Agreement)
print("\n2. Sıralama Tutarlılığı Değerlendirmesi (Ranking Agreement)")

# Her model için önerilen doküman indekslerini alma
model_doc_indices = {}

# TF-IDF modelleri için
tfidf_stemmed_indices = tfidf_results[tfidf_results['Model'] == 'TF-IDF Stemmed']['Index'].tolist()
tfidf_lemmatized_indices = tfidf_results[tfidf_results['Model'] == 'TF-IDF Lemmatized']['Index'].tolist()

model_doc_indices['tfidf_stemmed'] = set(tfidf_stemmed_indices)
model_doc_indices['tfidf_lemmatized'] = set(tfidf_lemmatized_indices)

# Word2Vec modelleri için
for model in all_results["model_results"].keys():
    indices = [item[0] for item in all_results["model_results"][model]]
    model_doc_indices[model] = set(indices)

# Jaccard benzerliği hesaplama
jaccard_matrix = {}
for model1 in all_models:
    jaccard_matrix[model1] = {}
    for model2 in all_models:
        # İki modelin önerdiği dokümanların kesişimi
        intersection = len(model_doc_indices[model1].intersection(model_doc_indices[model2]))
        # İki modelin önerdiği dokümanların birleşimi
        union = len(model_doc_indices[model1].union(model_doc_indices[model2]))
        
        # Jaccard benzerliği hesaplama
        jaccard = intersection / union if union > 0 else 0
        jaccard_matrix[model1][model2] = jaccard

# Jaccard matrisini DataFrame'e dönüştürme
jaccard_df = pd.DataFrame(jaccard_matrix)

# Jaccard matrisini görselleştirme
plt.figure(figsize=(14, 12))
sns.heatmap(jaccard_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Modeller Arası Jaccard Benzerliği")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "jaccard_similarity.png"))
print("Jaccard benzerliği ısı haritası kaydedildi.")

# Jaccard matrisini CSV dosyasına kaydetme
jaccard_df.to_csv(os.path.join(results_dir, "jaccard_similarity_matrix.csv"))
print("Jaccard benzerliği matrisi CSV dosyasına kaydedildi.")

# Subjektif değerlendirme sonuçlarını CSV dosyasına kaydetme
subjective_df = pd.DataFrame({
    "Model": list(subjective_scores.keys()),
    "Score1": [scores[0] for scores in subjective_scores.values()],
    "Score2": [scores[1] for scores in subjective_scores.values()],
    "Score3": [scores[2] for scores in subjective_scores.values()],
    "Score4": [scores[3] for scores in subjective_scores.values()],
    "Score5": [scores[4] for scores in subjective_scores.values()],
    "Average": [np.mean(scores) for scores in subjective_scores.values()]
})

subjective_df.to_csv(os.path.join(results_dir, "subjective_evaluation.csv"), index=False)
print("Anlamsal değerlendirme sonuçları CSV dosyasına kaydedildi.")

# Model yapılandırmalarının etkisini analiz etme
print("\n3. Model Yapılandırmalarının Etkisi")

# Word2Vec modellerini yapılandırma parametrelerine göre gruplandırma
model_configs = {}
for model in all_results["model_results"].keys():
    # Model adını parçalara ayırma
    parts = model.split('_')
    
    if len(parts) >= 4:
        model_type = parts[0]  # stemmed veya lemmatized
        algorithm = parts[1]   # cbow veya skipgram
        window = parts[2]      # win2 veya win4
        dim = parts[3]         # dim100 veya dim300
        
        # Yapılandırma bilgilerini kaydetme
        config = {
            "model_type": model_type,
            "algorithm": algorithm,
            "window": window,
            "dim": dim,
            "avg_score": model_avg_scores.get(model, 0)
        }
        
        model_configs[model] = config

# Yapılandırma parametrelerine göre ortalama puanları hesaplama
config_avg_scores = {
    "model_type": {},
    "algorithm": {},
    "window": {},
    "dim": {}
}

for model, config in model_configs.items():
    # Model türüne göre
    model_type = config["model_type"]
    if model_type not in config_avg_scores["model_type"]:
        config_avg_scores["model_type"][model_type] = []
    config_avg_scores["model_type"][model_type].append(config["avg_score"])
    
    # Algoritma türüne göre
    algorithm = config["algorithm"]
    if algorithm not in config_avg_scores["algorithm"]:
        config_avg_scores["algorithm"][algorithm] = []
    config_avg_scores["algorithm"][algorithm].append(config["avg_score"])
    
    # Pencere boyutuna göre
    window = config["window"]
    if window not in config_avg_scores["window"]:
        config_avg_scores["window"][window] = []
    config_avg_scores["window"][window].append(config["avg_score"])
    
    # Vektör boyutuna göre
    dim = config["dim"]
    if dim not in config_avg_scores["dim"]:
        config_avg_scores["dim"][dim] = []
    config_avg_scores["dim"][dim].append(config["avg_score"])

# Her parametre için ortalama puanları hesaplama
for param, scores in config_avg_scores.items():
    for value, score_list in scores.items():
        config_avg_scores[param][value] = np.mean(score_list)

# Yapılandırma parametrelerinin etkisini görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Model türü
sns.barplot(x=list(config_avg_scores["model_type"].keys()), 
            y=list(config_avg_scores["model_type"].values()),
            ax=axes[0, 0])
axes[0, 0].set_title("Model Türünün Etkisi")
axes[0, 0].set_ylabel("Ortalama Puan")

# Algoritma
sns.barplot(x=list(config_avg_scores["algorithm"].keys()), 
            y=list(config_avg_scores["algorithm"].values()),
            ax=axes[0, 1])
axes[0, 1].set_title("Algoritmanın Etkisi")
axes[0, 1].set_ylabel("Ortalama Puan")

# Pencere boyutu
sns.barplot(x=list(config_avg_scores["window"].keys()), 
            y=list(config_avg_scores["window"].values()),
            ax=axes[1, 0])
axes[1, 0].set_title("Pencere Boyutunun Etkisi")
axes[1, 0].set_ylabel("Ortalama Puan")

# Vektör boyutu
sns.barplot(x=list(config_avg_scores["dim"].keys()), 
            y=list(config_avg_scores["dim"].values()),
            ax=axes[1, 1])
axes[1, 1].set_title("Vektör Boyutunun Etkisi")
axes[1, 1].set_ylabel("Ortalama Puan")

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "config_parameters_effect.png"))
print("Yapılandırma parametrelerinin etkisi grafiği kaydedildi.")

# Yapılandırma parametrelerinin etkisini CSV dosyasına kaydetme
config_effect_df = pd.DataFrame({
    "Parameter": [],
    "Value": [],
    "Average_Score": []
})

for param, values in config_avg_scores.items():
    for value, score in values.items():
        config_effect_df = pd.concat([config_effect_df, pd.DataFrame({
            "Parameter": [param],
            "Value": [value],
            "Average_Score": [score]
        })], ignore_index=True)

config_effect_df.to_csv(os.path.join(results_dir, "config_parameters_effect.csv"), index=False)
print("Yapılandırma parametrelerinin etkisi CSV dosyasına kaydedildi.")

print("\nModel karşılaştırma ve değerlendirme tamamlandı.")
print("Sonuçlar ve grafikler 'results/' dizinine kaydedildi.")

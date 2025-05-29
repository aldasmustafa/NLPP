import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter
import json

# Grafik ayarları
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Dizinlerin var olduğundan emin olalım
raw_data_dir = "data/raw/"
processed_data_dir = "data/processed/"
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

print("Veri seti yükleniyor...")
# Excel dosyasını okuma
data = pd.read_excel('../data/bank.xlsx')

print(f"Veri seti boyutu: {data.shape}")
print(f"Sütunlar: {data.columns.tolist()}")

# TRANSACTION DETAILS sütununu çıkarma
transaction_details = data['TRANSACTION DETAILS'].dropna().astype(str)
print(f"Toplam işlem sayısı: {len(transaction_details)}")
print(f"Benzersiz işlem sayısı: {transaction_details.nunique()}")

# İlk 10 işlem detayını gösterme
print("\nÖrnek işlem detayları (ilk 10):")
for i, detail in enumerate(transaction_details.head(10)):
    print(f"{i+1}. {detail}")

# İşlem detaylarını metin dosyasına kaydetme
with open(os.path.join(raw_data_dir, "transaction_details.txt"), "w", encoding="utf-8") as f:
    for detail in transaction_details:
        f.write(detail + "\n")

print(f"\nİşlem detayları '{raw_data_dir}transaction_details.txt' dosyasına kaydedildi.")

# Tüm metinleri birleştirme
all_text = " ".join(transaction_details)

# Kelimelere ayırma (basit tokenization)
words = re.findall(r'\w+', all_text.lower())

# Kelime frekanslarını hesaplama
word_counts = Counter(words)

# En sık kullanılan 20 kelimeyi gösterme
print("\nEn sık kullanılan 20 kelime:")
for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")

# Zipf yasası grafiği için verileri hazırlama
word_freq = [(word, count) for word, count in word_counts.items()]
word_freq.sort(key=lambda x: x[1], reverse=True)

ranks = np.arange(1, len(word_freq) + 1)
frequencies = np.array([freq for word, freq in word_freq])

# Log-log grafiği çizme
plt.figure(figsize=(12, 8))
plt.loglog(ranks, frequencies, 'b.')
plt.xlabel('Kelime Sıralaması (log)', fontsize=14)
plt.ylabel('Kelime Frekansı (log)', fontsize=14)
plt.title('Zipf Yasası Analizi (Ham Veri)', fontsize=16)
plt.grid(True, alpha=0.3)

# Zipf yasası eğrisi (1/rank ilişkisi)
plt.loglog(ranks, frequencies[0] / ranks, 'r-', label='Zipf Yasası (1/rank)')
plt.legend()
plt.savefig(os.path.join(raw_data_dir, 'zipf_raw_data.png'), dpi=300, bbox_inches='tight')
print(f"\nZipf yasası grafiği '{raw_data_dir}zipf_raw_data.png' dosyasına kaydedildi.")

# Veri seti istatistikleri
total_words = len(words)
unique_words = len(word_counts)
vocabulary_richness = unique_words / total_words

print(f"\nVeri seti istatistikleri:")
print(f"Toplam kelime sayısı: {total_words}")
print(f"Benzersiz kelime sayısı: {unique_words}")
print(f"Kelime çeşitliliği oranı: {vocabulary_richness:.4f}")

# Veri seti yeterliliği değerlendirmesi
if total_words > 10000 and unique_words > 1000:
    print("Veri seti boyut olarak yeterlidir.")
else:
    print("Veri seti boyut olarak yetersiz olabilir.")

# Veri seti bilgilerini JSON dosyasına kaydetme
dataset_info = {
    "source": "Bank transaction details from bank.xlsx",
    "size": {
        "documents": len(transaction_details),
        "total_words": total_words,
        "unique_words": unique_words,
        "vocabulary_richness": vocabulary_richness
    },
    "format": "Text",
    "example": transaction_details.head(5).tolist()
}

with open(os.path.join(raw_data_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=4)

print(f"\nVeri seti bilgileri '{raw_data_dir}dataset_info.json' dosyasına kaydedildi.")

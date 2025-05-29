import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
os.makedirs(processed_data_dir, exist_ok=True)

# Stemming sonrası verileri yükleme
stemmed_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_data.csv"))

# Tüm kelimeleri birleştirme
stemmed_df['text'] = stemmed_df['text'].astype(str)
all_words = ' '.join(stemmed_df['text']).split()

# Kelime frekanslarını hesaplama
word_counts = Counter(all_words)

# En sık kullanılan 20 kelimeyi gösterme
print("Stemming sonrası en sık kullanılan 20 kelime:")
for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")

# Zipf yasası grafiği için verileri hazırlama
word_freq = [(word, count) for word, count in word_counts.items()]
word_freq.sort(key=lambda x: x[1], reverse=True)

ranks = np.arange(1, len(word_freq) + 1)
frequencies = np.array([freq for word, freq in word_freq])

# Log-log grafiği çizme
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, 'b.')
plt.xlabel('Kelime Sıralaması (log)', fontsize=12)
plt.ylabel('Kelime Frekansı (log)', fontsize=12)
plt.title('Zipf Yasası Analizi (Stemming Sonrası)', fontsize=14)
plt.grid(True, alpha=0.3)

# Zipf yasası eğrisi (1/rank ilişkisi)
plt.loglog(ranks, frequencies[0] / ranks, 'r-', label='Zipf Yasası (1/rank)')
plt.legend()
plt.savefig(os.path.join(processed_data_dir, 'zipf_stemmed_data.png'), dpi=300, bbox_inches='tight')
plt.show()

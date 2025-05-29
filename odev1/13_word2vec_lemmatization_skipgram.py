import pandas as pd
import os
import time
from gensim.models import Word2Vec

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
models_dir = "models/"
os.makedirs(models_dir, exist_ok=True)

# Lemmatization sonrası verileri yükleme
lemmatized_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"))
lemmatized_df['text'] = lemmatized_df['text'].astype(str)

# Verileri kelime listelerine dönüştürme
sentences = [text.split() for text in lemmatized_df['text']]

# Model parametreleri
model_type = 'skipgram'  # Skip-gram modeli
window = 4
vector_size = 300

# Model adını oluşturma
model_name = f"lemmatized_{model_type}_win{window}_dim{vector_size}"

# Eğitim başlangıç zamanı
start_time = time.time()

# Word2Vec modelini oluşturma ve eğitme
sg = 1  # sg=0: CBOW, sg=1: Skip-gram
model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, 
                 min_count=2, workers=4, sg=sg)

# Eğitim bitiş zamanı
end_time = time.time()
training_time = end_time - start_time

# Modeli kaydetme
model_path = os.path.join(models_dir, f"{model_name}.model")
model.save(model_path)

print(f"Model '{model_name}' başarıyla eğitildi ve kaydedildi.")
print(f"Eğitim süresi: {training_time:.2f} saniye")
print(f"Model boyutu: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
print(f"Kelime dağarcığı boyutu: {len(model.wv.key_to_index)} kelime")

# En sık kullanılan kelimelerden birini seçme
if len(model.wv.key_to_index) > 0:
    # Kelime frekanslarına göre sıralama
    example_word = sorted(model.wv.key_to_index.items(), key=lambda x: model.wv.get_vecattr(x[0], "count"), reverse=True)[0][0]
    
    try:
        # Örnek kelimeye en benzer 5 kelimeyi bulma
        similar_words = model.wv.most_similar(example_word, topn=5)
        print(f"\n'{example_word}' kelimesine en benzer 5 kelime:")
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
    except KeyError:
        print(f"'{example_word}' kelimesi modelin kelime dağarcığında bulunmuyor.")

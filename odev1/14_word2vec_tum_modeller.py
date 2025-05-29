import pandas as pd
import os
import time
from gensim.models import Word2Vec
from tqdm import tqdm

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
models_dir = "models/"
os.makedirs(models_dir, exist_ok=True)

# Stemming ve lemmatization sonrası verileri yükleme
stemmed_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_data.csv"))
lemmatized_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"))

# Verileri string'e dönüştürme
stemmed_df['text'] = stemmed_df['text'].astype(str)
lemmatized_df['text'] = lemmatized_df['text'].astype(str)

# Verileri kelime listelerine dönüştürme
stemmed_sentences = [text.split() for text in stemmed_df['text']]
lemmatized_sentences = [text.split() for text in lemmatized_df['text']]

# Word2Vec model parametreleri
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# Modelleri eğitme fonksiyonu
def train_models(sentences, params_list, prefix):
    models_info = []
    
    for params in tqdm(params_list, desc=f"{prefix} modelleri eğitiliyor"):
        model_type = params['model_type']
        window = params['window']
        vector_size = params['vector_size']
        
        # Model adını oluşturma
        model_name = f"{prefix}_{model_type}_win{window}_dim{vector_size}"
        
        # Eğitim başlangıç zamanı
        start_time = time.time()
        
        # Word2Vec modelini oluşturma ve eğitme
        sg = 1 if model_type == 'skipgram' else 0  # sg=0: CBOW, sg=1: Skip-gram
        model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, 
                         min_count=2, workers=4, sg=sg)
        
        # Eğitim bitiş zamanı
        end_time = time.time()
        training_time = end_time - start_time
        
        # Modeli kaydetme
        model_path = os.path.join(models_dir, f"{model_name}.model")
        model.save(model_path)
        
        # Model bilgilerini kaydetme
        model_info = {
            'model_name': model_name,
            'model_type': model_type,
            'window': window,
            'vector_size': vector_size,
            'training_time': training_time,
            'model_size': os.path.getsize(model_path) / (1024*1024),  # MB cinsinden
            'vocabulary_size': len(model.wv.key_to_index)
        }
        
        models_info.append(model_info)
        
    return models_info

# Stemming sonrası veriler için modelleri eğitme
print("Stemming sonrası veriler için Word2Vec modelleri eğitiliyor...")
stemmed_models_info = train_models(stemmed_sentences, parameters, "stemmed")

# Lemmatization sonrası veriler için modelleri eğitme
print("\nLemmatization sonrası veriler için Word2Vec modelleri eğitiliyor...")
lemmatized_models_info = train_models(lemmatized_sentences, parameters, "lemmatized")

# Model bilgilerini DataFrame'e dönüştürme ve kaydetme
stemmed_models_df = pd.DataFrame(stemmed_models_info)
lemmatized_models_df = pd.DataFrame(lemmatized_models_info)

# Model bilgilerini gösterme
print("\nStemming sonrası Word2Vec modelleri:")
print(stemmed_models_df[['model_name', 'model_type', 'window', 'vector_size', 'training_time', 'model_size', 'vocabulary_size']])

print("\nLemmatization sonrası Word2Vec modelleri:")
print(lemmatized_models_df[['model_name', 'model_type', 'window', 'vector_size', 'training_time', 'model_size', 'vocabulary_size']])

# Model bilgilerini CSV dosyalarına kaydetme
stemmed_models_df.to_csv(os.path.join(processed_data_dir, "stemmed_models_info.csv"), index=False)
lemmatized_models_df.to_csv(os.path.join(processed_data_dir, "lemmatized_models_info.csv"), index=False)

print(f"\nModel bilgileri '{processed_data_dir}stemmed_models_info.csv' ve '{processed_data_dir}lemmatized_models_info.csv' dosyalarına kaydedildi.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import time
from tqdm import tqdm

# Grafik ayarları
plt.style.use('ggplot')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
models_dir = "models/"
os.makedirs(models_dir, exist_ok=True)

print("İşlenmiş verileri yüklüyorum...")
# Stemming ve lemmatization sonucu elde edilen verileri yükleme
stemmed_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_data.csv"))
lemmatized_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_data.csv"))

print(f"Stemming sonrası veri boyutu: {stemmed_df.shape}")
print(f"Lemmatization sonrası veri boyutu: {lemmatized_df.shape}")

# TF-IDF Vektörleştirme
print("\nTF-IDF vektörleştirme işlemi yapılıyor...")

# Stemming sonrası veriler için TF-IDF vektörleştirme
print("Stemming sonrası veriler için TF-IDF vektörleştirme...")
# TF-IDF vektörleştirici oluşturma
tfidf_vectorizer_stemmed = TfidfVectorizer(max_features=5000)  # En sık kullanılan 5000 kelimeyi kullan

# Vektörleştirme işlemini gerçekleştirme
tfidf_matrix_stemmed = tfidf_vectorizer_stemmed.fit_transform(stemmed_df['text'])

# Özellikleri (kelimeleri) alma
feature_names_stemmed = tfidf_vectorizer_stemmed.get_feature_names_out()

# TF-IDF matrisini DataFrame'e dönüştürme
tfidf_df_stemmed = pd.DataFrame(tfidf_matrix_stemmed.toarray(), columns=feature_names_stemmed)
tfidf_df_stemmed.index = stemmed_df['document_id']

# TF-IDF DataFrame'ini kaydetme
tfidf_df_stemmed.to_csv(os.path.join(processed_data_dir, "tfidf_stemmed.csv"))

print(f"Stemming sonrası TF-IDF matris boyutu: {tfidf_df_stemmed.shape}")

# Lemmatization sonrası veriler için TF-IDF vektörleştirme
print("Lemmatization sonrası veriler için TF-IDF vektörleştirme...")
# TF-IDF vektörleştirici oluşturma
tfidf_vectorizer_lemmatized = TfidfVectorizer(max_features=5000)  # En sık kullanılan 5000 kelimeyi kullan

# Vektörleştirme işlemini gerçekleştirme
tfidf_matrix_lemmatized = tfidf_vectorizer_lemmatized.fit_transform(lemmatized_df['text'])

# Özellikleri (kelimeleri) alma
feature_names_lemmatized = tfidf_vectorizer_lemmatized.get_feature_names_out()

# TF-IDF matrisini DataFrame'e dönüştürme
tfidf_df_lemmatized = pd.DataFrame(tfidf_matrix_lemmatized.toarray(), columns=feature_names_lemmatized)
tfidf_df_lemmatized.index = lemmatized_df['document_id']

# TF-IDF DataFrame'ini kaydetme
tfidf_df_lemmatized.to_csv(os.path.join(processed_data_dir, "tfidf_lemmatized.csv"))

print(f"Lemmatization sonrası TF-IDF matris boyutu: {tfidf_df_lemmatized.shape}")

# TF-IDF Sonuçlarının İncelenmesi
print("\nTF-IDF sonuçlarının incelenmesi...")

# Her belge için en yüksek TF-IDF değerine sahip 5 kelimeyi bulma
def get_top_tfidf_words(tfidf_df, n=5):
    results = []
    for i in range(min(5, len(tfidf_df))):  # İlk 5 belgeyi incele
        row = tfidf_df.iloc[i]
        top_indices = row.nlargest(n).index
        top_values = row.nlargest(n).values
        results.append({
            'document_id': tfidf_df.index[i],
            'top_words': [(word, value) for word, value in zip(top_indices, top_values)]
        })
    return results

# Stemming sonrası en yüksek TF-IDF değerine sahip kelimeleri gösterme
top_stemmed_words = get_top_tfidf_words(tfidf_df_stemmed)

print("Stemming sonrası en yüksek TF-IDF değerine sahip kelimeler:")
for i, result in enumerate(top_stemmed_words):
    print(f"Belge {result['document_id']} için en önemli kelimeler:")
    for word, value in result['top_words']:
        print(f"  {word}: {value:.4f}")
    print()

# Lemmatization sonrası en yüksek TF-IDF değerine sahip kelimeleri gösterme
top_lemmatized_words = get_top_tfidf_words(tfidf_df_lemmatized)

print("Lemmatization sonrası en yüksek TF-IDF değerine sahip kelimeler:")
for i, result in enumerate(top_lemmatized_words):
    print(f"Belge {result['document_id']} için en önemli kelimeler:")
    for word, value in result['top_words']:
        print(f"  {word}: {value:.4f}")
    print()

# Word2Vec için verileri hazırlama
print("\nWord2Vec için verileri hazırlama...")
# Stemming sonrası verileri kelime listelerine dönüştürme
stemmed_sentences = [text.split() for text in stemmed_df['text']]

# Lemmatization sonrası verileri kelime listelerine dönüştürme
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

# Word2Vec modellerini eğitme fonksiyonu
def train_word2vec_models(sentences, params_list, prefix):
    models_info = []
    
    for params in tqdm(params_list, desc=f"Training {prefix} models"):
        model_type = params['model_type']
        window = params['window']
        vector_size = params['vector_size']
        
        # Model adını oluşturma
        model_name = f"{prefix}_{model_type}_win{window}_dim{vector_size}"
        
        # Eğitim başlangıç zamanı
        start_time = time.time()
        
        # Word2Vec modelini oluşturma ve eğitme
        sg = 1 if model_type == 'skipgram' else 0  # sg=1: Skip-gram, sg=0: CBOW
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
            'vocabulary_size': len(model.wv.key_to_index),
            'model_path': model_path
        }
        
        models_info.append(model_info)
        
    return models_info

# Stemming sonrası veriler için Word2Vec modellerini eğitme
print("\nStemming sonrası veriler için Word2Vec modellerini eğitiyorum...")
stemmed_models_info = train_word2vec_models(stemmed_sentences, parameters, "stemmed")

# Lemmatization sonrası veriler için Word2Vec modellerini eğitme
print("\nLemmatization sonrası veriler için Word2Vec modellerini eğitiyorum...")
lemmatized_models_info = train_word2vec_models(lemmatized_sentences, parameters, "lemmatized")

# Word2Vec Model Bilgilerinin Gösterilmesi
print("\nWord2Vec model bilgileri:")

# Stemming sonrası Word2Vec model bilgilerini gösterme
stemmed_models_df = pd.DataFrame(stemmed_models_info)
print("Stemming sonrası Word2Vec modelleri:")
print(stemmed_models_df[['model_name', 'model_type', 'window', 'vector_size', 'training_time', 'model_size', 'vocabulary_size']])

# Lemmatization sonrası Word2Vec model bilgilerini gösterme
lemmatized_models_df = pd.DataFrame(lemmatized_models_info)
print("\nLemmatization sonrası Word2Vec modelleri:")
print(lemmatized_models_df[['model_name', 'model_type', 'window', 'vector_size', 'training_time', 'model_size', 'vocabulary_size']])

# Model bilgilerini CSV dosyalarına kaydetme
stemmed_models_df.to_csv(os.path.join(processed_data_dir, "stemmed_models_info.csv"), index=False)
lemmatized_models_df.to_csv(os.path.join(processed_data_dir, "lemmatized_models_info.csv"), index=False)
print(f"\nModel bilgileri '{processed_data_dir}stemmed_models_info.csv' ve '{processed_data_dir}lemmatized_models_info.csv' dosyalarına kaydedildi.")

# Word2Vec Modellerinin Benzerlik Analizi
print("\nWord2Vec modellerinin benzerlik analizi yapılıyor...")

# Benzerlik analizi için örnek kelime seçme
# Veri setindeki en sık kullanılan kelimelerden birini seçelim
example_word = None

# Stemming sonrası modeller için benzerlik analizi
print("Stemming sonrası modeller için benzerlik analizi:")
for model_info in stemmed_models_info:
    model_path = model_info['model_path']
    model = Word2Vec.load(model_path)
    
    print(f"\nModel: {model_info['model_name']}")
    
    if example_word is None or example_word not in model.wv.key_to_index:
        # Modelin kelime dağarcığından en sık kullanılan kelimeyi seçme
        if len(model.wv.key_to_index) > 0:
            # Kelime frekanslarına göre sıralama
            example_word = sorted(model.wv.key_to_index.items(), key=lambda x: model.wv.get_vecattr(x[0], "count"), reverse=True)[0][0]
    
    try:
        # Örnek kelimeye en benzer 5 kelimeyi bulma
        similar_words = model.wv.most_similar(example_word, topn=5)
        print(f"'{example_word}' kelimesine en benzer 5 kelime:")
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
    except KeyError:
        print(f"'{example_word}' kelimesi modelin kelime dağarcığında bulunmuyor.")
        # Alternatif bir kelime seçme
        try:
            # Modelin kelime dağarcığından rastgele bir kelime seçme
            alt_word = list(model.wv.key_to_index.keys())[0]
            similar_words = model.wv.most_similar(alt_word, topn=5)
            print(f"Alternatif olarak '{alt_word}' kelimesine en benzer 5 kelime:")
            for word, similarity in similar_words:
                print(f"  {word}: {similarity:.4f}")
        except:
            print("Benzerlik analizi yapılamadı.")

# Lemmatization sonrası modeller için benzerlik analizi
print("\nLemmatization sonrası modeller için benzerlik analizi:")
for model_info in lemmatized_models_info:
    model_path = model_info['model_path']
    model = Word2Vec.load(model_path)
    
    print(f"\nModel: {model_info['model_name']}")
    
    if example_word is None or example_word not in model.wv.key_to_index:
        # Modelin kelime dağarcığından en sık kullanılan kelimeyi seçme
        if len(model.wv.key_to_index) > 0:
            # Kelime frekanslarına göre sıralama
            example_word = sorted(model.wv.key_to_index.items(), key=lambda x: model.wv.get_vecattr(x[0], "count"), reverse=True)[0][0]
    
    try:
        # Örnek kelimeye en benzer 5 kelimeyi bulma
        similar_words = model.wv.most_similar(example_word, topn=5)
        print(f"'{example_word}' kelimesine en benzer 5 kelime:")
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
    except KeyError:
        print(f"'{example_word}' kelimesi modelin kelime dağarcığında bulunmuyor.")
        # Alternatif bir kelime seçme
        try:
            # Modelin kelime dağarcığından rastgele bir kelime seçme
            alt_word = list(model.wv.key_to_index.keys())[0]
            similar_words = model.wv.most_similar(alt_word, topn=5)
            print(f"Alternatif olarak '{alt_word}' kelimesine en benzer 5 kelime:")
            for word, similarity in similar_words:
                print(f"  {word}: {similarity:.4f}")
        except:
            print("Benzerlik analizi yapılamadı.")

# Model Başarısının Değerlendirilmesi
print("\nModel Başarısının Değerlendirilmesi:")
print("\nStemming vs Lemmatization:")
print("- Stemming, kelimeleri köklerine indirgerken anlamsal bilgileri kaybedebilir.")
print("- Lemmatization, kelimeleri anlamsal köklerine indirgediği için daha anlamlı sonuçlar verebilir.")

print("\nCBOW vs Skip-gram:")
print("- CBOW, bir kelimenin bağlamından (çevresindeki kelimelerden) o kelimeyi tahmin etmeye çalışır.")
print("- Skip-gram, bir kelimeden o kelimenin bağlamını (çevresindeki kelimeleri) tahmin etmeye çalışır.")
print("- Skip-gram genellikle nadir kelimeler için daha iyi performans gösterir, ancak eğitim süresi daha uzundur.")

print("\nPencere Boyutu (Window Size):")
print("- Küçük pencere boyutu (2), yakın kelimelere daha fazla önem verir ve sözdizimsel ilişkileri daha iyi yakalar.")
print("- Büyük pencere boyutu (4), daha geniş bağlamı dikkate alır ve anlamsal ilişkileri daha iyi yakalar.")

print("\nVektör Boyutu (Vector Size):")
print("- Küçük vektör boyutu (100), eğitim süresini kısaltır ancak temsil kapasitesi sınırlıdır.")
print("- Büyük vektör boyutu (300), daha zengin temsiller sağlar ancak eğitim süresi daha uzundur ve aşırı öğrenme riski vardır.")

print("\nBeklenen En Başarılı Model:")
print("Lemmatization sonrası, Skip-gram, pencere boyutu 4 ve vektör boyutu 300 olan modelin en başarılı olması beklenir.")
print("Çünkü:")
print("- Lemmatization, anlamsal bilgileri koruduğu için daha iyi sonuçlar verir.")
print("- Skip-gram, nadir kelimeler için daha iyi performans gösterir.")
print("- Büyük pencere boyutu, daha geniş bağlamı dikkate alır.")
print("- Büyük vektör boyutu, daha zengin temsiller sağlar.")

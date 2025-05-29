import os
import pandas as pd
from gensim.models import Word2Vec

# Dizinlerin var olduğundan emin olalım
models_dir = "models/"
os.makedirs(models_dir, exist_ok=True)

# Model bilgilerini yükleme
processed_data_dir = "data/processed/"
try:
    stemmed_models_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_models_info.csv"))
    lemmatized_models_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_models_info.csv"))
    model_info_loaded = True
except FileNotFoundError:
    print("Model bilgileri bulunamadı. Önce 14_word2vec_tum_modeller.py dosyasını çalıştırın.")
    model_info_loaded = False

# Eğer model bilgileri yüklendiyse benzerlik analizini yap
if model_info_loaded:
    # Stemming sonrası modeller için benzerlik analizi
    print("Stemming sonrası modeller için benzerlik analizi:")
    for _, row in stemmed_models_df.iterrows():
        model_name = row['model_name']
        model_path = os.path.join(models_dir, f"{model_name}.model")
        
        try:
            model = Word2Vec.load(model_path)
            print(f"\nModel: {model_name}")
            
            # En sık kullanılan kelimeyi bulma
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
            else:
                print("Model kelime dağarcığı boş.")
        except FileNotFoundError:
            print(f"Model dosyası bulunamadı: {model_path}")
    
    # Lemmatization sonrası modeller için benzerlik analizi
    print("\nLemmatization sonrası modeller için benzerlik analizi:")
    for _, row in lemmatized_models_df.iterrows():
        model_name = row['model_name']
        model_path = os.path.join(models_dir, f"{model_name}.model")
        
        try:
            model = Word2Vec.load(model_path)
            print(f"\nModel: {model_name}")
            
            # En sık kullanılan kelimeyi bulma
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
            else:
                print("Model kelime dağarcığı boş.")
        except FileNotFoundError:
            print(f"Model dosyası bulunamadı: {model_path}")
else:
    print("Benzerlik analizi yapılamadı. Önce modelleri eğitin.")

import pandas as pd
import os

# Dizinlerin var olduğundan emin olalım
processed_data_dir = "data/processed/"
models_dir = "models/"

# Model değerlendirmesi
print("Model Başarısının Değerlendirilmesi:")
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

# Model bilgilerini yükleme ve karşılaştırma
try:
    stemmed_models_df = pd.read_csv(os.path.join(processed_data_dir, "stemmed_models_info.csv"))
    lemmatized_models_df = pd.read_csv(os.path.join(processed_data_dir, "lemmatized_models_info.csv"))
    
    # En iyi modeli bulma (eğitim süresi ve kelime dağarcığı boyutuna göre)
    best_stemmed_model = stemmed_models_df.sort_values(by=['vocabulary_size', 'training_time'], ascending=[False, True]).iloc[0]
    best_lemmatized_model = lemmatized_models_df.sort_values(by=['vocabulary_size', 'training_time'], ascending=[False, True]).iloc[0]
    
    print("\nEn İyi Stemming Modeli:")
    print(f"Model Adı: {best_stemmed_model['model_name']}")
    print(f"Model Tipi: {best_stemmed_model['model_type']}")
    print(f"Pencere Boyutu: {best_stemmed_model['window']}")
    print(f"Vektör Boyutu: {best_stemmed_model['vector_size']}")
    print(f"Kelime Dağarcığı Boyutu: {best_stemmed_model['vocabulary_size']}")
    print(f"Eğitim Süresi: {best_stemmed_model['training_time']:.2f} saniye")
    
    print("\nEn İyi Lemmatization Modeli:")
    print(f"Model Adı: {best_lemmatized_model['model_name']}")
    print(f"Model Tipi: {best_lemmatized_model['model_type']}")
    print(f"Pencere Boyutu: {best_lemmatized_model['window']}")
    print(f"Vektör Boyutu: {best_lemmatized_model['vector_size']}")
    print(f"Kelime Dağarcığı Boyutu: {best_lemmatized_model['vocabulary_size']}")
    print(f"Eğitim Süresi: {best_lemmatized_model['training_time']:.2f} saniye")
    
except FileNotFoundError:
    print("\nModel bilgileri bulunamadı. Önce modelleri eğitin.")

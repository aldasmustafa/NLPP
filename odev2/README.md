# Doğal Dil İşleme Dersi - Ödev 2

## Eğitilen Modellerle Metin Benzerliği Hesaplama ve Değerlendirme

Bu ödev, birinci ödevde ön işleme tabi tutulan verilerle eğitilen Word2Vec ve TF-IDF modellerini kullanarak metinler arası benzerlik hesaplamaları yapmayı amaçlamaktadır. Aynı zamanda kullanılan modellerin karşılaştırmalı başarımı değerlendirilecektir.

## Dosya Yapısı

- `data/processed/`: İşlenmiş veri dosyaları
  - `stemmed_data.csv`: Stemming uygulanmış veri seti
  - `lemmatized_data.csv`: Lemmatization uygulanmış veri seti
  - `tfidf_stemmed.csv`: Stemming sonrası TF-IDF matrisi
  - `tfidf_lemmatized.csv`: Lemmatization sonrası TF-IDF matrisi
  - `stemmed_models_info.csv`: Stemming modelleri bilgileri
  - `lemmatized_models_info.csv`: Lemmatization modelleri bilgileri

- `models/`: Eğitilmiş Word2Vec modelleri
  - 8 adet stemming sonrası model
  - 8 adet lemmatization sonrası model

- Python Dosyaları:
  - `1_tfidf_benzerlik.py`: TF-IDF modelleri ile benzerlik hesaplama
  - `2_word2vec_benzerlik.py`: Word2Vec modelleri ile benzerlik hesaplama
  - `3_model_karsilastirma.py`: Modellerin karşılaştırılması ve değerlendirilmesi

## Çalıştırma Talimatları

1. TF-IDF benzerlik hesaplama:
   ```
   python 1_tfidf_benzerlik.py
   ```

2. Word2Vec benzerlik hesaplama:
   ```
   python 2_word2vec_benzerlik.py
   ```

3. Model karşılaştırma ve değerlendirme:
   ```
   python 3_model_karsilastirma.py
   ```

## Proje Yapısı

[Ödev 2 için proje yapısı buraya eklenecek]

[Gerekli kütüphaneler ve kurulum talimatları buraya eklenecek]

## Nasıl Çalıştırılır

[Çalıştırma talimatları buraya eklenecek]

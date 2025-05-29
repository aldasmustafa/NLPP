# Doğal Dil İşleme Dersi - Ödev 1

Bu repo, Doğal Dil İşleme dersi için hazırlanan Ödev 1'i içermektedir. Ödevde metin tabanlı bir veri seti üzerinde ön işleme adımları uygulanmış ve vektörleştirme yapılmıştır.

## Proje Yapısı

```
odev1/
  ├── data/
  │   ├── raw/         # Ham veri dosyaları
  │   └── processed/   # İşlenmiş veri dosyaları
  ├── notebooks/       # Jupyter notebook dosyaları
  └── models/          # Eğitilen modeller
```

## Veri Seti

[Veri seti hakkında bilgi buraya eklenecek]

## Kurulum

Projeyi çalıştırmak için gerekli kütüphaneler:

```bash
pip install -r requirements.txt
```

## Adımlar

1. Veri seti indirme ve hazırlama
2. Ön işleme adımları (Tokenization, Stop word removal, Lowercasing, Lemmatization, Stemming)
3. Zipf Yasası Analizi
4. Vektörleştirme (TF-IDF ve Word2Vec)
5. Sonuçların değerlendirilmesi

## Nasıl Çalıştırılır

Notebooks klasöründeki Jupyter notebook dosyalarını sırasıyla çalıştırın:

1. `1_data_preparation.ipynb`: Veri setini indirme ve hazırlama
2. `2_preprocessing.ipynb`: Ön işleme adımları
3. `3_vectorization.ipynb`: Vektörleştirme işlemleri

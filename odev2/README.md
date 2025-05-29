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

```
odev2/
  ├── data/
  │   ├── processed/       # İşlenmiş veri dosyaları
  │   │   ├── lemmatized_data.csv
  │   │   ├── stemmed_data.csv
  │   │   └── ...
  ├── models/             # Eğitilmiş Word2Vec modelleri
  │   ├── lemmatized_cbow_win2_dim100.model
  │   ├── stemmed_skipgram_win4_dim300.model
  │   └── ...
  ├── results/            # Analiz sonuçları ve grafikler
  │   ├── tfidf_similarity_results.csv
  │   ├── word2vec_similarity_results.csv
  │   ├── jaccard_similarity.png
  │   └── ...
  ├── 1_tfidf_benzerlik.py           # TF-IDF benzerlik analizi
  ├── 2_word2vec_benzerlik.py        # Word2Vec benzerlik analizi
  ├── 3_model_karsilastirma.py       # Model karşılaştırma
  ├── benzerlik_analizi_raporu.md    # Detaylı analiz raporu
  ├── benzerlik_analizi_ozet.md      # Özet rapor
  ├── tfidf_benzerlik_sonuclari.md   # TF-IDF sonuçları
  ├── word2vec_stemming_sonuclari.md # Stemming sonuçları
  ├── word2vec_lemmatization_sonuclari.md # Lemmatization sonuçları
  └── word2vec_id_sonuclari.md       # ID formatında sonuçlar
```

### Gerekli Kütüphaneler

Projeyi çalıştırmak için aşağıdaki kütüphanelerin kurulu olması gerekmektedir:

```bash
pip install pandas numpy scikit-learn gensim matplotlib nltk seaborn
```

Ayrıca NLTK için gerekli kaynakları indirmek için:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Nasıl Çalıştırılır

1. Öncelikle veri setinin işlenmiş olduğundan emin olun. Eğer Ödev 1'i tamamladıysanız, işlenmiş veriler zaten hazırdır.

2. TF-IDF benzerlik analizi için:
   ```bash
   python 1_tfidf_benzerlik.py
   ```
   Bu script, TF-IDF vektörleri arasındaki benzerliği hesaplar ve sonuçları `tfidf_benzerlik_sonuclari.md` dosyasına kaydeder.

3. Word2Vec benzerlik analizi için:
   ```bash
   python 2_word2vec_benzerlik.py
   ```
   Bu script, Word2Vec modelleri kullanarak metin benzerliklerini hesaplar ve sonuçları farklı formatlarda kaydeder:
   - `word2vec_stemming_sonuclari.md`: Stemming sonuçları
   - `word2vec_lemmatization_sonuclari.md`: Lemmatization sonuçları
   - `word2vec_id_sonuclari.md`: ID formatında sonuçlar

4. Model karşılaştırma ve değerlendirme için:
   ```bash
   python 3_model_karsilastirma.py
   ```
   Bu script, farklı modellerin performansını karşılaştırır ve sonuçları `model_degerlendirme_sonuclari.md` dosyasına kaydeder.

5. Tüm sonuçları incelemek için `benzerlik_analizi_raporu.md` dosyasını açabilirsiniz. Bu dosya, tüm analizlerin detaylı bir raporunu içerir ve PDF formatına dönüştürülebilir.

# Metin Benzerlik Analizi Raporu

## Özet

Bu rapor, NLP ödev 2 kapsamında gerçekleştirilen metin benzerlik analizi çalışmalarının sonuçlarını içermektedir. Çalışmada, TF-IDF ve Word2Vec modelleri kullanılarak banka işlem detayları üzerinde benzerlik hesaplamaları yapılmıştır. Örnek metin olarak "irctc corpor offic ac" kullanılmıştır.

## 1. TF-IDF Benzerlik Analizi Sonuçları

### Stemming Sonrası TF-IDF Benzerliği

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.3789 | vaishanvi ac |
| 2 | 0.3789 | vaishnavi ac |
| 3 | 0.3694 | oasi corpor train |
| 4 | 0.3694 | pahuja corpor |
| 5 | 0.3219 | neft 000028094933 irctc c |

### Lemmatization Sonrası TF-IDF Benzerliği

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.3781 | oasis corporate training |
| 2 | 0.3775 | vaishanvi ac |
| 3 | 0.3775 | vaishnavi ac |
| 4 | 0.3207 | neft 000028094933 irctc c |
| 5 | 0.3207 | neft 000040875666 irctc c |

## 2. Word2Vec Benzerlik Analizi Sonuçları

### Stemming Modelleri

#### Model: stemmed_cbow_win2_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9889 | indian oil corpor lt |
| 2 | 0.9889 | indian oil corpor lt |
| 3 | 0.9889 | indian oil corpor lt |
| 4 | 0.9889 | indian oil corpor lt |
| 5 | 0.9889 | indian oil corpor lt |

#### Model: stemmed_skipgram_win2_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9915 | bidderboy corpor |
| 2 | 0.9915 | bidderboy corpor |
| 3 | 0.9915 | bidderboy corpor |
| 4 | 0.9915 | bidderboy corpor |
| 5 | 0.9915 | bidderboy corpor |

#### Model: stemmed_cbow_win4_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9874 | bidderboy corpor |
| 2 | 0.9874 | bidderboy corpor |
| 3 | 0.9874 | bidderboy corpor |
| 4 | 0.9874 | bidderboy corpor |
| 5 | 0.9874 | bidderboy corpor |

#### Model: stemmed_skipgram_win4_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9907 | bidderboy corpor |
| 2 | 0.9907 | bidderboy corpor |
| 3 | 0.9907 | bidderboy corpor |
| 4 | 0.9907 | bidderboy corpor |
| 5 | 0.9907 | bidderboy corpor |

#### Model: stemmed_cbow_win2_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9839 | indian oil corpor lt |
| 2 | 0.9839 | indian oil corpor lt |
| 3 | 0.9839 | indian oil corpor lt |
| 4 | 0.9839 | indian oil corpor lt |
| 5 | 0.9839 | indian oil corpor lt |

#### Model: stemmed_skipgram_win2_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9908 | bidderboy corpor |
| 2 | 0.9908 | bidderboy corpor |
| 3 | 0.9908 | bidderboy corpor |
| 4 | 0.9908 | bidderboy corpor |
| 5 | 0.9908 | bidderboy corpor |

#### Model: stemmed_cbow_win4_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9916 | bidderboy corpor |
| 2 | 0.9916 | bidderboy corpor |
| 3 | 0.9916 | bidderboy corpor |
| 4 | 0.9916 | bidderboy corpor |
| 5 | 0.9916 | bidderboy corpor |

#### Model: stemmed_skipgram_win4_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9920 | bidderboy corpor |
| 2 | 0.9920 | bidderboy corpor |
| 3 | 0.9920 | bidderboy corpor |
| 4 | 0.9920 | bidderboy corpor |
| 5 | 0.9920 | bidderboy corpor |

### Lemmatization Modelleri

#### Model: lemmatized_cbow_win2_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9612 | vaishnavi ac |
| 2 | 0.9590 | vaishanvi ac |
| 3 | 0.9440 | oasis corporate training |
| 4 | 0.8672 | neft 000028094933 irctc c |
| 5 | 0.8672 | neft 000040875666 irctc c |

#### Model: lemmatized_skipgram_win2_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9840 | vaishnavi ac |
| 2 | 0.9824 | vaishanvi ac |
| 3 | 0.9586 | oasis corporate training |
| 4 | 0.8883 | neft 000028094933 irctc c |
| 5 | 0.8883 | neft 000040875666 irctc c |

#### Model: lemmatized_cbow_win4_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9864 | vaishnavi ac |
| 2 | 0.9857 | vaishanvi ac |
| 3 | 0.9820 | oasis corporate training |
| 4 | 0.8891 | neft 000028094933 irctc c |
| 5 | 0.8891 | neft 000040875666 irctc c |

#### Model: lemmatized_skipgram_win4_dim100

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9878 | vaishnavi ac |
| 2 | 0.9876 | vaishanvi ac |
| 3 | 0.9624 | oasis corporate training |
| 4 | 0.8975 | neft 000028094933 irctc c |
| 5 | 0.8975 | neft 000040875666 irctc c |

#### Model: lemmatized_cbow_win2_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9801 | vaishnavi ac |
| 2 | 0.9786 | vaishanvi ac |
| 3 | 0.9643 | oasis corporate training |
| 4 | 0.8801 | neft 000028094933 irctc c |
| 5 | 0.8801 | neft 000040875666 irctc c |

#### Model: lemmatized_skipgram_win2_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9792 | vaishnavi ac |
| 2 | 0.9788 | vaishanvi ac |
| 3 | 0.9620 | oasis corporate training |
| 4 | 0.8893 | neft 000028094933 irctc c |
| 5 | 0.8893 | neft 000040875666 irctc c |

#### Model: lemmatized_cbow_win4_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9874 | vaishnavi ac |
| 2 | 0.9873 | vaishanvi ac |
| 3 | 0.9637 | oasis corporate training |
| 4 | 0.9181 | vaishanvi air conditioner |
| 5 | 0.8968 | neft 000028094933 irctc c |

#### Model: lemmatized_skipgram_win4_dim300

| Sıra | Benzerlik Skoru | Metin |
|------|-----------------|-------|
| 1 | 0.9910 | vaishnavi ac |
| 2 | 0.9901 | vaishanvi ac |
| 3 | 0.9737 | oasis corporate training |
| 4 | 0.8933 | neft 000028094933 irctc c |
| 5 | 0.8933 | neft 000040875666 irctc c |

## 3. Model Değerlendirme Sonuçları

### 3.1 Anlamsal Değerlendirme (Subjective Evaluation)

| Model | Puan 1 | Puan 2 | Puan 3 | Puan 4 | Puan 5 | Ortalama |
|-------|--------|--------|--------|--------|--------|----------|
| tfidf_stemmed | 5 | 5 | 5 | 5 | 5 | 5.0 |
| tfidf_lemmatized | 5 | 5 | 5 | 5 | 5 | 5.0 |
| stemmed_cbow_win2_dim100 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| stemmed_skipgram_win2_dim100 | 5 | 5 | 4 | 4 | 4 | 4.4 |
| stemmed_cbow_win4_dim100 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| stemmed_skipgram_win4_dim100 | 5 | 5 | 4 | 4 | 4 | 4.4 |
| stemmed_cbow_win2_dim300 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| stemmed_skipgram_win2_dim300 | 5 | 5 | 4 | 4 | 4 | 4.4 |
| stemmed_cbow_win4_dim300 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| stemmed_skipgram_win4_dim300 | 5 | 5 | 4 | 4 | 4 | 4.4 |
| lemmatized_cbow_win2_dim100 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| lemmatized_skipgram_win2_dim100 | 5 | 5 | 4 | 4 | 4 | 4.4 |
| lemmatized_cbow_win4_dim100 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| lemmatized_skipgram_win4_dim100 | 5 | 5 | 4 | 4 | 4 | 4.4 |
| lemmatized_cbow_win2_dim300 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| lemmatized_skipgram_win2_dim300 | 5 | 5 | 4 | 4 | 4 | 4.4 |
| lemmatized_cbow_win4_dim300 | 5 | 4 | 4 | 4 | 4 | 4.2 |
| lemmatized_skipgram_win4_dim300 | 5 | 5 | 4 | 4 | 4 | 4.4 |

### 3.2 Model Performans Sıralaması (Ortalama Puanlara Göre)

1. **tfidf_stemmed**: 5.00
2. **tfidf_lemmatized**: 5.00
3. **stemmed_skipgram_win2_dim100**: 4.40
4. **stemmed_skipgram_win4_dim100**: 4.40
5. **stemmed_skipgram_win2_dim300**: 4.40
6. **stemmed_skipgram_win4_dim300**: 4.40
7. **lemmatized_skipgram_win2_dim100**: 4.40
8. **lemmatized_skipgram_win4_dim100**: 4.40
9. **lemmatized_skipgram_win2_dim300**: 4.40
10. **lemmatized_skipgram_win4_dim300**: 4.40
11. **stemmed_cbow_win2_dim100**: 4.20
12. **stemmed_cbow_win4_dim100**: 4.20
13. **stemmed_cbow_win2_dim300**: 4.20
14. **stemmed_cbow_win4_dim300**: 4.20
15. **lemmatized_cbow_win2_dim100**: 4.20
16. **lemmatized_cbow_win4_dim100**: 4.20
17. **lemmatized_cbow_win2_dim300**: 4.20
18. **lemmatized_cbow_win4_dim300**: 4.20

## 4. Değerlendirme ve Sonuçlar

### 4.1 TF-IDF vs Word2Vec Karşılaştırması

- **TF-IDF Modelleri**: Kısa metinlerde benzerlik analizinde en yüksek performansı gösterdi (ortalama puan: 5.00/5.00)
- **Word2Vec Modelleri**: Skip-gram modelleri (ortalama puan: 4.40/5.00), CBOW modellerinden (ortalama puan: 4.20/5.00) daha iyi performans gösterdi

### 4.2 Stemming vs Lemmatization Karşılaştırması

- **Stemming**: Özellikle TF-IDF modellerinde yüksek performans gösterdi
- **Lemmatization**: Word2Vec modellerinde daha tutarlı sonuçlar üretti, özellikle "vaishnavi ac", "vaishanvi ac" ve "oasis corporate training" metinlerini başarıyla buldu

### 4.3 Model Parametrelerinin Etkisi

- **Pencere Boyutu (Window Size)**: Pencere boyutunun 2'den 4'e çıkarılması, benzerlik skorlarında küçük artışlara neden oldu
- **Vektör Boyutu (Dimension)**: Vektör boyutunun 100'den 300'e çıkarılması, bazı modellerde performansı artırırken bazılarında düşürdü
- **Algoritma Seçimi**: Skip-gram algoritması, CBOW'a göre daha tutarlı ve yüksek benzerlik skorları üretti

### 4.4 Jaccard Benzerliği Analizi

Jaccard benzerliği analizi, modellerin sıralama tutarlılığını ölçmek için kullanıldı. Analiz sonuçlarına göre:

- TF-IDF modelleri kendi aralarında 0.67 Jaccard benzerliği gösterdi
- Lemmatization TF-IDF modeli, lemmatization Word2Vec modelleriyle 1.00 Jaccard benzerliği gösterdi
- Stemming Word2Vec modelleri kendi aralarında yüksek benzerlik gösterdi (1.00)
- Lemmatization Word2Vec modelleri kendi aralarında yüksek benzerlik gösterdi (1.00)
- Stemming ve lemmatization modelleri arasında düşük benzerlik görüldü (0.00)

## 5. Genel Sonuç

Bu çalışmada, banka işlem detayları üzerinde TF-IDF ve Word2Vec modelleri kullanılarak metin benzerlik analizi gerçekleştirilmiştir. Sonuçlar, TF-IDF modellerinin kısa metinlerde benzerlik analizinde yüksek performans gösterdiğini, Word2Vec modellerinde ise Skip-gram yaklaşımının CBOW'dan daha başarılı olduğunu ortaya koymuştur. Ayrıca, model parametrelerinin (pencere boyutu, vektör boyutu) etkisi, model türüne göre değişkenlik göstermiştir.

Bu analiz, NLP uygulamalarında metin benzerliği hesaplama için farklı modellerin performansını değerlendirme ve karşılaştırma konusunda önemli bilgiler sunmaktadır.

# Metin Benzerlik Analizi Özet Raporu

## Özet

Bu rapor, NLP ödev 2 kapsamında gerçekleştirilen metin benzerlik analizi çalışmalarının sonuçlarını içermektedir. Çalışmada, TF-IDF ve Word2Vec modelleri kullanılarak banka işlem detayları üzerinde benzerlik hesaplamaları yapılmıştır. Örnek metin olarak "irctc corpor offic ac" kullanılmıştır.

## Model Performans Sıralaması (Ortalama Puanlara Göre)

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

## Değerlendirme ve Sonuçlar

### TF-IDF vs Word2Vec Karşılaştırması

- **TF-IDF Modelleri**: Kısa metinlerde benzerlik analizinde en yüksek performansı gösterdi (ortalama puan: 5.00/5.00)
- **Word2Vec Modelleri**: Skip-gram modelleri (ortalama puan: 4.40/5.00), CBOW modellerinden (ortalama puan: 4.20/5.00) daha iyi performans gösterdi

### Stemming vs Lemmatization Karşılaştırması

- **Stemming**: Özellikle TF-IDF modellerinde yüksek performans gösterdi
- **Lemmatization**: Word2Vec modellerinde daha tutarlı sonuçlar üretti, özellikle "vaishnavi ac", "vaishanvi ac" ve "oasis corporate training" metinlerini başarıyla buldu

### Model Parametrelerinin Etkisi

- **Pencere Boyutu (Window Size)**: Pencere boyutunun 2'den 4'e çıkarılması, benzerlik skorlarında küçük artışlara neden oldu
- **Vektör Boyutu (Dimension)**: Vektör boyutunun 100'den 300'e çıkarılması, bazı modellerde performansı artırırken bazılarında düşürdü
- **Algoritma Seçimi**: Skip-gram algoritması, CBOW'a göre daha tutarlı ve yüksek benzerlik skorları üretti

## Genel Sonuç

Bu çalışmada, banka işlem detayları üzerinde TF-IDF ve Word2Vec modelleri kullanılarak metin benzerlik analizi gerçekleştirilmiştir. Sonuçlar, TF-IDF modellerinin kısa metinlerde benzerlik analizinde yüksek performans gösterdiğini, Word2Vec modellerinde ise Skip-gram yaklaşımının CBOW'dan daha başarılı olduğunu ortaya koymuştur. Ayrıca, model parametrelerinin (pencere boyutu, vektör boyutu) etkisi, model türüne göre değişkenlik göstermiştir.

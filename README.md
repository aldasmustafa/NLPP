# Doğal Dil İşleme Projeleri

Bu repo, Doğal Dil İşleme dersi için hazırlanan ödevleri içermektedir.

## Proje Yapısı

```
nlpp/
  ├── odev1/          # Ödev 1: Vektörleştirme ve Ön İşleme
  ├── odev2/          # Ödev 2: Benzerlik Analizi
  └── data/           # Ortak veri dosyaları
```

## Ödev 1: Vektörleştirme ve Ön İşleme

Ödev 1'de metin tabanlı bir veri seti üzerinde ön işleme adımları uygulanmış ve vektörleştirme yapılmıştır.

### Adımlar

1. Veri seti indirme ve hazırlama
2. Ön işleme adımları (Tokenization, Stop word removal, Lowercasing, Lemmatization, Stemming)
3. Zipf Yasası Analizi
4. Vektörleştirme (TF-IDF ve Word2Vec)
5. Sonuçların değerlendirilmesi

## Ödev 2: Benzerlik Analizi

Ödev 2'de metin benzerlik analizi yapılmıştır. Sonuçlar hem metin hem de ID formatında gösterilmiştir.

### Özellikler

- Word2Vec ile benzerlik analizi
- Sonuçların bölümlere ayrılarak gösterilmesi
- Markdown formatında rapor oluşturma
- PDF çıktı alma

## Kurulum

Projeyi çalıştırmak için gerekli kütüphaneler:

```bash
pip install -r requirements.txt
```

## Nasıl Çalıştırılır

Her bir ödev klasöründeki Python dosyalarını sırasıyla çalıştırın.

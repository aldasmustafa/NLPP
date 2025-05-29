import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup

# HTML etiketlerini temizleme fonksiyonu
def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Özel karakterleri ve noktalama işaretlerini temizleme fonksiyonu
def clean_special_chars(text):
    # Noktalama işaretlerini kaldırma
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    # Fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization fonksiyonu
def tokenize_text(text):
    return word_tokenize(text)

# Küçük harfe dönüştürme fonksiyonu
def lowercase_text(tokens):
    return [token.lower() for token in tokens]

# Stop word removal fonksiyonu
def remove_stopwords(tokens, lang='english'):
    stop_words = set(stopwords.words(lang))
    return [token for token in tokens if token not in stop_words]

# Stemming fonksiyonu
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# Lemmatization fonksiyonu
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Örnek kullanım
text = "This is an example text with some HTML <b>tags</b> and punctuation marks!"
print("Orijinal metin:", text)

# Adım adım ön işleme
cleaned_html = clean_html(text)
print("HTML temizleme sonrası:", cleaned_html)

cleaned_special = clean_special_chars(cleaned_html)
print("Özel karakterler temizlendikten sonra:", cleaned_special)

tokens = tokenize_text(cleaned_special)
print("Tokenization sonrası:", tokens)

lowercase_tokens = lowercase_text(tokens)
print("Küçük harfe dönüştürme sonrası:", lowercase_tokens)

no_stopwords = remove_stopwords(lowercase_tokens)
print("Stop word removal sonrası:", no_stopwords)

stemmed_tokens = stem_tokens(no_stopwords)
print("Stemming sonrası:", stemmed_tokens)

lemmatized_tokens = lemmatize_tokens(no_stopwords)
print("Lemmatization sonrası:", lemmatized_tokens)

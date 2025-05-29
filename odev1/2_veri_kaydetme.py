import pandas as pd
import os

# Dizinlerin var olduğundan emin olalım
raw_data_dir = "data/raw/"
os.makedirs(raw_data_dir, exist_ok=True)

# Veri setini yükleme
data = pd.read_excel('../data/bank.xlsx')

# TRANSACTION DETAILS sütununu çıkarma
transaction_details = data['TRANSACTION DETAILS'].dropna().astype(str)

# İşlem detaylarını metin dosyasına kaydetme
with open(os.path.join(raw_data_dir, "transaction_details.txt"), "w", encoding="utf-8") as f:
    for detail in transaction_details:
        f.write(detail + "\n")

print(f"İşlem detayları '{raw_data_dir}transaction_details.txt' dosyasına kaydedildi.")

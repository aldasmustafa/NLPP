import pandas as pd

# Veri setini yükleme
data = pd.read_excel('../data/bank.xlsx')

# Veri seti hakkında genel bilgiler
print(f"Veri seti boyutu: {data.shape}")
print(f"Sütunlar: {data.columns.tolist()}")
print("\nİlk 5 satır:")
print(data.head())

# TRANSACTION DETAILS sütununu inceleme
transaction_details = data['TRANSACTION DETAILS'].dropna().astype(str)
print(f"\nToplam işlem sayısı: {len(transaction_details)}")
print(f"Benzersiz işlem sayısı: {transaction_details.nunique()}")
print("\nÖrnek işlem detayları (ilk 10):")
for i, detail in enumerate(transaction_details.head(10)):
    print(f"{i+1}. {detail}")

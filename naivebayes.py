import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Muat dataset
data = pd.read_csv('annual-enterprise-survey-2023-financial-year-provisional.csv')

# Bersihkan data (hapus nilai yang hilang)
data_cleaned = data.dropna()

# Label encoding untuk fitur-fitur kategorikal
label_encoder = LabelEncoder()

data_cleaned['Industry_code_NZSIOC'] = label_encoder.fit_transform(data_cleaned['Industry_code_NZSIOC'])
data_cleaned['Industry_name_NZSIOC'] = label_encoder.fit_transform(data_cleaned['Industry_name_NZSIOC'])
data_cleaned['Variable_code'] = label_encoder.fit_transform(data_cleaned['Variable_code'])
data_cleaned['Variable_name'] = label_encoder.fit_transform(data_cleaned['Variable_name'])

# Pastikan 'Value' adalah numerik
data_cleaned['Value'] = pd.to_numeric(data_cleaned['Value'], errors='coerce')
data_cleaned['Value'] = data_cleaned['Value'].fillna(data_cleaned['Value'].mean())

# Pendekatan 1: Klasifikasi berdasarkan Variable_category
X_var_category = data_cleaned[['Industry_code_NZSIOC', 'Variable_code', 'Value']]
y_var_category = label_encoder.fit_transform(data_cleaned['Variable_category'])

# Split data
X_train_var, X_test_var, y_train_var, y_test_var = train_test_split(X_var_category, y_var_category, test_size=0.3, random_state=42)

# Model Naive Bayes untuk klasifikasi Variable_category
model_var_category = GaussianNB()
model_var_category.fit(X_train_var, y_train_var)

# Prediksi dan evaluasi
y_pred_var = model_var_category.predict(X_test_var)
accuracy_var_category = accuracy_score(y_test_var, y_pred_var)
print(f'Akurasi klasifikasi Variable_category: {accuracy_var_category:.2f}')

# Pendekatan 2: Klasifikasi berdasarkan Industry_name_NZSIOC
X_industry = data_cleaned[['Variable_code', 'Value']]
y_industry = data_cleaned['Industry_name_NZSIOC']

# Split data
X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(X_industry, y_industry, test_size=0.3, random_state=42)

# Model Naive Bayes untuk klasifikasi Industry_name_NZSIOC
model_industry = GaussianNB()
model_industry.fit(X_train_ind, y_train_ind)

# Prediksi dan evaluasi
y_pred_ind = model_industry.predict(X_test_ind)
accuracy_industry = accuracy_score(y_test_ind, y_pred_ind)
print(f'Akurasi klasifikasi Industry_name_NZSIOC: {accuracy_industry:.2f}')

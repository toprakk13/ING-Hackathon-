import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ===== 1️⃣ Verileri oku =====
df_customer_history = pd.read_csv("customer_history.csv")
df_customer = pd.read_csv("customers.csv")
df_referance = pd.read_csv("referance_data.csv")  # eğitim cust_id’leri
df_test_ref = pd.read_csv("referance_data_test.csv")  # test cust_id’leri

# Tarih sütunu datetime formatına çevir
df_customer_history['date'] = pd.to_datetime(df_customer_history['date'])

# ===== 2️⃣ Feature engineering =====
latest_history = df_customer_history.sort_values('date').groupby('cust_id').tail(1)
train_df = df_referance.merge(df_customer, on='cust_id', how='left')
train_df = train_df.merge(latest_history, on='cust_id', how='left')

# Eksik değerleri doldur
num_cols = ['mobile_eft_all_cnt', 'mobile_eft_all_amt', 'cc_transaction_all_cnt', 'cc_transaction_all_amt']
train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
train_df['work_sector'] = train_df['work_sector'].fillna('Unknown')

# Tenure grupları
bins = [0, 6, 12, 24, 60, 120]
labels = ['0-6m', '6-12m', '1-2y', '2-5y', '5y+']
train_df['tenure_group'] = pd.cut(train_df['tenure'], bins=bins, labels=labels)

# Yeni features: oranlar
train_df['mobile_eft_avg_amt'] = train_df['mobile_eft_all_amt'] / (train_df['mobile_eft_all_cnt'] + 1)
train_df['cc_avg_amt'] = train_df['cc_transaction_all_amt'] / (train_df['cc_transaction_all_cnt'] + 1)

# Kategorik encode
cat_cols = ['gender', 'work_type', 'work_sector', 'province', 'tenure_group']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    encoders[col] = le

# Feature ve target
X_train = train_df.drop(['cust_id', 'date', 'ref_date', 'religion', 'churn'], axis=1)
y_train = train_df['churn']

# ===== 3️⃣ Modeli eğit =====
model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# ===== 4️⃣ Test setini hazırla ve tahmin yap =====
test_df = df_customer[df_customer['cust_id'].isin(df_test_ref['cust_id'])].merge(
    latest_history, on='cust_id', how='left'
)
test_df[num_cols] = test_df[num_cols].fillna(train_df[num_cols].median())
test_df['work_sector'] = test_df['work_sector'].fillna('Unknown')
test_df['tenure_group'] = pd.cut(test_df['tenure'], bins=bins, labels=labels)
test_df['mobile_eft_avg_amt'] = test_df['mobile_eft_all_amt'] / (test_df['mobile_eft_all_cnt'] + 1)
test_df['cc_avg_amt'] = test_df['cc_transaction_all_amt'] / (test_df['cc_transaction_all_cnt'] + 1)

for col in cat_cols:
    le = encoders[col]
    test_df[col] = test_df[col].astype(str)
    test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
    if 'Unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'Unknown')
    test_df[col] = le.transform(test_df[col])

X_test = test_df[X_train.columns]

# Tahmin olasılıkları
y_test_prob = model.predict_proba(X_test)[:, 1]




# ===== 5️⃣ Submission =====
submission_df = pd.DataFrame({
    'cust_id': test_df['cust_id'],
    'churn': y_test_prob
})

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
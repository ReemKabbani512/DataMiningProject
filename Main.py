# مشروع توصية المنتجات باستخدام قواعد الارتباط والعنقدة

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import sqlite3
import streamlit as st

# ===================================================
# 1. قراءة البيانات
# ===================================================

# قراءة بيانات الفواتير
df_invoices = pd.read_csv("Invoices_Dataset_for_Association_Rules.csv")
# قراءة بيانات المنتجات
df_products = pd.read_csv("Extended_Products_Dataset__25_Products_ (1).csv")

# ===================================================
# 2. استكشاف البيانات (EDA)
# ===================================================

print("معلومات الفواتير:")
print(df_invoices.info())
print(df_invoices.describe())

print("معلومات المنتجات:")
print(df_products.info())
print(df_products.describe())

# ===================================================
# 3. تنظيف البيانات
# ===================================================

# التحقق من القيم المفقودة
print("Missing values in products:\n", df_products.isnull().sum())

# تعويض القيم الفارغة بالمتوسط
df_products.fillna(df_products.mean(numeric_only=True), inplace=True)

# إزالة التكرارات إن وجدت
df_products.drop_duplicates(inplace=True)
df_invoices.drop_duplicates(inplace=True)

# ===================================================
# 4. تجهيز البيانات لقواعد الارتباط
# ===================================================

transactions = df_invoices.groupby('InvoiceID')['ProductID'].apply(list).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# ===================================================
# 5. تطبيق Apriori واستخراج قواعد الارتباط
# ===================================================

frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules.sort_values(by='lift', ascending=False)

# ===================================================
# 6. إعداد البيانات للعَنقدة (مع وبدون السعر)
# ===================================================

features_with_price = ['Price', 'Rating', 'Stock', 'WarrantyYears', 'WeightKg', 'VolumeCm3', 'PowerWatt']
features_without_price = ['Rating', 'Stock', 'WarrantyYears', 'WeightKg', 'VolumeCm3', 'PowerWatt']

scaler = StandardScaler()

# مع السعر
X_with_price = scaler.fit_transform(df_products[features_with_price])
kmeans_with_price = KMeans(n_clusters=4, random_state=42)
df_products['Cluster_WithPrice'] = kmeans_with_price.fit_predict(X_with_price)

# بدون السعر
X_without_price = scaler.fit_transform(df_products[features_without_price])
kmeans_without_price = KMeans(n_clusters=4, random_state=42)
df_products['Cluster_WithoutPrice'] = kmeans_without_price.fit_predict(X_without_price)

# ===================================================
# 7. تخزين النتائج في قاعدة بيانات SQLite
# ===================================================

conn = sqlite3.connect('products_clusters.db')
cursor = conn.cursor()
#اذا الجدول غير موجود أسسلي ياه
cursor.execute("CREATE TABLE IF NOT EXISTS products ( \
    ProductID INTEGER PRIMARY KEY, \
    ProductName TEXT, \
    Category TEXT, \
    Brand TEXT, \
    SupplierCountry TEXT, \
    ConnectivityType TEXT, \
    MaterialType TEXT, \
    UsageType TEXT, \
    PriceCategory TEXT, \
    Price REAL, \
    Rating REAL, \
    Stock INTEGER, \
    WarrantyYears INTEGER, \
    WeightKg REAL, \
    VolumeCm3 REAL, \
    PowerWatt REAL, \
    Cluster_WithPrice INTEGER, \
    Cluster_WithoutPrice INTEGER \
)")
conn.commit()

for _, row in df_products.iterrows():
    cursor.execute(
        "INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            row['ProductID'],
            row['ProductName'],
            row['Category'],
            row['Brand'],
            row['SupplierCountry'],
            row['ConnectivityType'],
            row['MaterialType'],
            row['UsageType'],
            row['PriceCategory'],
            row['Price'],
            row['Rating'],
            row['Stock'],
            row['WarrantyYears'],
            row['WeightKg'],
            row['VolumeCm3'],
            row['PowerWatt'],
            int(row['Cluster_WithPrice']),
            int(row['Cluster_WithoutPrice'])
        )
    )
conn.commit()
conn.close()

# ===================================================
# 8. واجهة Streamlit لعرض توصيات المنتجات
# ===================================================

st.title("🔍 نظام توصية المنتجات الذكي")

conn = sqlite3.connect('products_clusters.db')
product_names = pd.read_sql_query("SELECT ProductName FROM products", conn)['ProductName'].tolist()
selected_product = st.selectbox("اختر منتجًا:", product_names)

product_data = pd.read_sql_query(f"SELECT * FROM products WHERE ProductName='{selected_product}'", conn)
cluster_label = product_data['Cluster_WithPrice'].values[0]

st.write(f" هذا المنتج ينتمي إلى العنقود رقم: {cluster_label} (باحتساب السعر)")

recommended = pd.read_sql_query(f"""
SELECT ProductName FROM products 
WHERE Cluster_WithPrice={cluster_label} AND ProductName != '{selected_product}'
""", conn)

st.subheader("📦 منتجات مشابهة لهذا المنتج:")
st.write(recommended)
conn.close()


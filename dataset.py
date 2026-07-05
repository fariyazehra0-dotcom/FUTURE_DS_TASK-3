# ============================================================
# TASK 3: MARKETING FUNNEL & CONVERSION PERFORMANCE ANALYSIS
# DATASET: BANK MARKETING (bank-full.csv)
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv('bank-full.csv', sep=';')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates:")
print(df.shape)

# ============================================================
# 3. BASIC EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nNumerical Statistics")
print(df.describe())

print("\nTarget Variable Distribution")
print(df['y'].value_counts())

# ============================================================
# 4. MARKETING FUNNEL ANALYSIS
# ============================================================

total_contacted = len(df)

converted = len(df[df['y'] == 'yes'])

not_converted = len(df[df['y'] == 'no'])

conversion_rate = (converted / total_contacted) * 100

drop_off_rate = (not_converted / total_contacted) * 100

print("\n" + "="*60)
print("MARKETING FUNNEL")
print("="*60)

print("Total Contacted Customers :", total_contacted)
print("Converted Customers       :", converted)
print("Not Converted             :", not_converted)
print("Conversion Rate (%)       :", round(conversion_rate,2))
print("Drop-Off Rate (%)         :", round(drop_off_rate,2))

# Funnel Visualization

funnel = pd.DataFrame({
    'Stage':['Contacted','Converted'],
    'Count':[total_contacted, converted]
})

plt.figure(figsize=(8,5))
sns.barplot(data=funnel, x='Stage', y='Count')
plt.title('Marketing Funnel')
plt.show()

# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(6,4))

sns.countplot(data=df, x='y')

plt.title("Conversion Distribution")
plt.xlabel("Subscribed")
plt.ylabel("Count")

plt.show()

# ============================================================
# 6. CONTACT CHANNEL PERFORMANCE
# ============================================================

contact_conversion = (
    df.groupby('contact')['y']
    .apply(lambda x: (x=='yes').mean()*100)
    .sort_values(ascending=False)
)

print("\nContact Channel Conversion Rates")
print(contact_conversion)

plt.figure(figsize=(8,5))

contact_conversion.plot(kind='bar')

plt.title("Conversion Rate by Contact Method")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 7. MONTH PERFORMANCE ANALYSIS
# ============================================================

month_conversion = (
    df.groupby('month')['y']
    .apply(lambda x:(x=='yes').mean()*100)
    .sort_values(ascending=False)
)

print("\nMonth Conversion Rates")
print(month_conversion)

plt.figure(figsize=(10,5))

month_conversion.plot(kind='bar')

plt.title("Conversion Rate by Month")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 8. JOB SEGMENT ANALYSIS
# ============================================================

job_conversion = (
    df.groupby('job')['y']
    .apply(lambda x:(x=='yes').mean()*100)
    .sort_values(ascending=False)
)

print("\nJob Conversion Rates")
print(job_conversion)

plt.figure(figsize=(12,6))

job_conversion.plot(kind='bar')

plt.title("Conversion Rate by Job")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 9. EDUCATION ANALYSIS
# ============================================================

education_conversion = (
    df.groupby('education')['y']
    .apply(lambda x:(x=='yes').mean()*100)
    .sort_values(ascending=False)
)

print("\nEducation Conversion Rates")
print(education_conversion)

plt.figure(figsize=(8,5))

education_conversion.plot(kind='bar')

plt.title("Conversion Rate by Education")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 10. MARITAL STATUS ANALYSIS
# ============================================================

marital_conversion = (
    df.groupby('marital')['y']
    .apply(lambda x:(x=='yes').mean()*100)
    .sort_values(ascending=False)
)

print("\nMarital Status Conversion Rates")
print(marital_conversion)

plt.figure(figsize=(8,5))

marital_conversion.plot(kind='bar')

plt.title("Conversion Rate by Marital Status")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 11. HOUSING LOAN ANALYSIS
# ============================================================

housing_conversion = (
    df.groupby('housing')['y']
    .apply(lambda x:(x=='yes').mean()*100)
)

print("\nHousing Loan Conversion")
print(housing_conversion)

plt.figure(figsize=(6,4))

housing_conversion.plot(kind='bar')

plt.title("Housing Loan Impact")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 12. PERSONAL LOAN ANALYSIS
# ============================================================

loan_conversion = (
    df.groupby('loan')['y']
    .apply(lambda x:(x=='yes').mean()*100)
)

print("\nPersonal Loan Conversion")
print(loan_conversion)

plt.figure(figsize=(6,4))

loan_conversion.plot(kind='bar')

plt.title("Personal Loan Impact")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 13. PREVIOUS CAMPAIGN ANALYSIS
# ============================================================

previous_campaign = (
    df.groupby('poutcome')['y']
    .apply(lambda x:(x=='yes').mean()*100)
    .sort_values(ascending=False)
)

print("\nPrevious Campaign Results")
print(previous_campaign)

plt.figure(figsize=(8,5))

previous_campaign.plot(kind='bar')

plt.title("Previous Campaign Outcome")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 14. AGE GROUP ANALYSIS
# ============================================================

df['Age_Group'] = pd.cut(
    df['age'],
    bins=[18,30,40,50,60,100],
    labels=['18-30','31-40','41-50','51-60','60+']
)

age_conversion = (
    df.groupby('Age_Group')['y']
    .apply(lambda x:(x=='yes').mean()*100)
)

print("\nAge Group Conversion")
print(age_conversion)

plt.figure(figsize=(8,5))

age_conversion.plot(kind='bar')

plt.title("Conversion Rate by Age Group")
plt.ylabel("Conversion %")

plt.show()

# ============================================================
# 15. NUMERIC CORRELATION HEATMAP
# ============================================================

numeric_cols = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================================
# 16. TOP PERFORMING SEGMENTS
# ============================================================

best_contact = contact_conversion.idxmax()
best_contact_rate = contact_conversion.max()

best_month = month_conversion.idxmax()
best_month_rate = month_conversion.max()

best_job = job_conversion.idxmax()
best_job_rate = job_conversion.max()

print("\n" + "="*60)
print("TOP PERFORMERS")
print("="*60)

print(f"Best Contact Method : {best_contact} ({best_contact_rate:.2f}%)")
print(f"Best Month          : {best_month} ({best_month_rate:.2f}%)")
print(f"Best Job Segment    : {best_job} ({best_job_rate:.2f}%)")

# ============================================================
# 17. FINAL REPORT
# ============================================================

print("\n" + "="*60)
print("FINAL BUSINESS REPORT")
print("="*60)

print(f"Total Customers Contacted : {total_contacted}")
print(f"Converted Customers       : {converted}")
print(f"Conversion Rate           : {conversion_rate:.2f}%")
print(f"Drop-Off Rate             : {drop_off_rate:.2f}%")

print("\nBUSINESS INSIGHTS")
print("- Most customers did not convert, indicating a large funnel drop-off.")
print("- Contact method significantly affects conversion rates.")
print("- Some months perform substantially better than others.")
print("- Certain job categories have higher conversion potential.")
print("- Previous successful campaign customers are more likely to convert again.")

print("\nRECOMMENDATIONS")
print("1. Focus marketing budget on the best-performing contact channels.")
print("2. Run campaigns during high-converting months.")
print("3. Retarget customers with previous successful responses.")
print("4. Personalize offers for high-converting customer segments.")
print("5. Improve campaign messaging for low-converting groups.")

print("\nAnalysis Completed Successfully!")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
chunk_size = 10000
chunks = []


for chunk in pd.read_csv("Retail_Transactions_Dataset.csv", chunksize=chunk_size):
    chunks.append(chunk)

df = pd.concat(chunks)

print(df.info())

print(df['Customer_Name'].value_counts())

print(df['Promotion'].value_counts(dropna=False))

print(df['Discount_Applied'].value_counts())

df["Promotion"] = df["Promotion"].fillna("None")

df['Date'] = pd.to_datetime(df['Date'])

customer_df = df.groupby("Customer_Name")["Total_Cost"].sum().reset_index()
print(customer_df)

cust_first_date_df = df.groupby("Customer_Name")["Date"].min().reset_index()
cust_first_date_df.rename(columns={"Date": "First_Transaction_Date"}, inplace=True)
print(cust_first_date_df)

cust_last_date_df = df.groupby("Customer_Name")["Date"].max().reset_index()
cust_last_date_df.rename(columns={"Date": "Last_Transaction_Date"}, inplace=True)

print(cust_last_date_df)

cust_items_df = df.groupby("Customer_Name")["Total_Items"].sum().reset_index()
print(cust_items_df)

cust_transactions_df = df.groupby("Customer_Name")["Total_Items"].count().reset_index()
cust_transactions_df.rename(columns={"Total_Items": "Number_of_Transactions"}, inplace=True)
print(cust_transactions_df)

merged_df = pd.merge(customer_df, cust_first_date_df,how="inner", on="Customer_Name")

merged_df = pd.merge(merged_df, cust_last_date_df, how="inner", on="Customer_Name")
print(merged_df)

merged_df = pd.merge(merged_df, cust_items_df, how="inner", on="Customer_Name")
print(merged_df)

merged_df = pd.merge(merged_df, cust_transactions_df, how="inner", on="Customer_Name")
print(merged_df)

merged_df["Items_per_transaction"] = merged_df["Total_Items"]/merged_df["Number_of_Transactions"]
merged_df["Average_Spend"] = merged_df["Total_Cost"]/merged_df["Number_of_Transactions"]
merged_df["Recency"] = (merged_df["Last_Transaction_Date"]-merged_df["First_Transaction_Date"]).dt.days

print(merged_df)

most_recent_date = merged_df["Last_Transaction_Date"].max()

merged_df["day_diff"] = (most_recent_date - merged_df["First_Transaction_Date"]).dt.days
merged_df["day_diff"] = merged_df["day_diff"].replace(0, 1)
merged_df["Avg_Daily_Spend"] = merged_df["Total_Cost"]/merged_df["day_diff"]
print(merged_df)
merged_df["Avg_Daily_Spend"].hist(bins=2000)
plt.xlim([0, 3])
plt.xlabel("Average Daily Spend")
plt.ylabel("Frequency")
plt.title("Distribution of Average Daily Spend")
plt.show()

merged_df["Average_Spend"].hist(bins=50)
plt.xlabel("Average Spend")
plt.ylabel("Frequency")
plt.title("Distribution of Average Spend per Transaction")
plt.show()


merged_df["Total_Cost"].hist(bins=2000)
plt.xlabel("Current Total Customer Spend")
plt.xlim([0, 1500])
plt.ylabel("Frequency")
plt.title("Distribution of Customer Spending")
plt.show()

corr = merged_df[["Total_Cost", "Average_Spend", "Number_of_Transactions",
                  "Avg_Daily_Spend", "Recency", "day_diff"]].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="viridis")
plt.title("Pearson's Correlation Matrix of Variables")
plt.show()


sns.scatterplot(merged_df[merged_df["Total_Cost"] < 5000], x="Recency", y="Total_Cost")
plt.xlabel("Recency")
plt.ylabel("Total Spend")
plt.title("Total Spend vs Recency")
plt.show()

merged_df['R_score'] = pd.cut(merged_df['Recency'], 4, labels=False) + 1
merged_df['F_score'] = pd.cut(merged_df['Number_of_Transactions'], 4, labels=False) + 1
merged_df['M_score'] = pd.cut(merged_df['Total_Cost'], 4, labels=False) + 1


merged_df['RFM_score'] = merged_df['R_score'].astype(str) + merged_df['F_score'].astype(str) + merged_df['M_score'].astype(str)


def segment_customer(df):
    if df['RFM_score'] == '111':
        return 'Low-Value Customers'
    elif df['RFM_score'] == '444':
        return 'High-Value Customers'
    else:
        return 'Mid-Value Customers'

merged_df['Segment'] = merged_df.apply(segment_customer, axis=1)
print(merged_df[['Customer_Name', 'RFM_score', 'Segment']].head())


customer_lifespan_years = (merged_df["Recency"].mean())/365
print(customer_lifespan_years)
merged_df['CLV'] = merged_df['Average_Spend'] * merged_df['Number_of_Transactions'] * customer_lifespan_years

print(merged_df[['Customer_Name', 'Segment', 'CLV']].head())

print(merged_df['RFM_score'].value_counts())

plt.figure(figsize=(10, 6))
merged_df.groupby("Segment")["CLV"].mean().plot(kind="bar")
plt.xlabel("Customer Segment", fontsize=6)
plt.xticks(rotation=15)
plt.ylabel("CLV ($)")
plt.yscale("log")
plt.title("Average Customer Lifetime Value by Segment")
plt.show()

plt.figure(figsize =(10, 6))
merged_df.groupby("Segment")["CLV"].sum().plot(kind="bar")
plt.xlabel("Customer Segment", fontsize=6)
plt.xticks(rotation=15)
plt.ylabel("CLV ($)")
plt.yscale("log")
plt.title("Sum of Customer Lifetime Value by Segment")
plt.show()

merged_df.to_csv("clv_summary.csv", index=False)


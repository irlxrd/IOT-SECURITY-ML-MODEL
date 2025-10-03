# unsw_training_eda.py
# Exploratory Data Analysis on UNSW-NB15 training set



import pandas as pd
import matplotlib.pyplot as plt




# 1. Load dataset
file_path = "/Users/igor/Downloads/UNSW_NB15_training-set.csv"

df = pd.read_csv(file_path)




print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist()[:10], "...")
print("\nFirst 5 rows:\n", df.head())


print("\nLabel distribution (0=benign, 1=attack):\n", df["label"].value_counts())
print("\nAttack categories:\n", df["attack_cat"].value_counts())





# Attack vs Benign counts
df["label"].value_counts().plot(kind="bar")
plt.title("Benign vs Attack")

plt.xticks([0,1], ["Attack", "Benign"], rotation=0)
plt.ylabel("Count")
plt.show()






# Attack categories
df["attack_cat"].value_counts().plot(kind="bar")
plt.title("Attack Categories")


plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()





# Source bytes (log scale boxplot)
plt.figure(figsize=(8,5))
data = [df[df["label"] == 0]["sbytes"], df[df["label"] == 1]["sbytes"]]
plt.boxplot(data, tick_labels=["Benign", "Attack"], showfliers=False)
plt.yscale("log")


plt.title("Source Bytes (log scale)")
plt.show()




# Protocol usage (top 10 only)
plt.figure(figsize=(10,6))
top_protocols = df["proto"].value_counts().nlargest(10).index
filtered = df[df["proto"].isin(top_protocols)]


protocol_counts = filtered.groupby(["proto", "label"]).size().unstack(fill_value=0)


protocol_counts.plot(kind="bar", stacked=True, figsize=(10,6))

plt.title("Top 10 Protocols (Benign vs Attack)")
plt.ylabel("Count")


plt.xticks(rotation=45)
plt.show()

import pandas as pd

df = pd.read_csv("fake_reviews_dataset.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns)

print("\nLabel Distribution:")
print(df["label"].value_counts())

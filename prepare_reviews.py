import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("fake_reviews_dataset.csv")

# Keep only needed columns
df = df[["text_", "label"]]

# Rename columns
df = df.rename(columns={"text_": "review_text"})

# Convert labels to numeric
df["label"] = df["label"].map({"OR": 0, "CG": 1})

# Drop nulls & duplicates
df = df.dropna()
df = df.drop_duplicates(subset=["review_text"])

# Remove very short reviews (<5 words)
df["word_count"] = df["review_text"].apply(lambda x: len(str(x).split()))
df = df[df["word_count"] >= 5]
df = df.drop(columns=["word_count"])

print("Final shape:", df.shape)
print(df["label"].value_counts())

# Split data
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

print("\nTrain:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

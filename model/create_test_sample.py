import pandas as pd

# Load full dataset
df = pd.read_csv("dataset/weatherAUS.csv")

# Drop rows with missing target
df = df.dropna(subset=["RainTomorrow"])

# Take random sample (200 rows)
test_sample = df.sample(n=200, random_state=42)

# Save to project root
test_sample.to_csv("dataset/test_sample.csv", index=False)

print("test_sample.csv created successfully!")

import pandas as pd

df = pd.read_csv("keypoints.csv")

print(f"✅ Total rows (frames): {len(df)}")
print(f"✅ Total columns (keypoints): {len(df.columns)}")
print("\nFirst 3 rows preview:")
print(df.head(3))
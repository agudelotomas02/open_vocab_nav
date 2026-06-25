import json
from collections import Counter

# Load semantic map
with open("semantic_map.json", "r") as f:
    data = json.load(f)

object_counts = Counter()

# Count objects across all frames
for frame, detections in data.items():

    for obj, count in detections.items():
        object_counts[obj] += count

print("\n=== Semantic Map Analysis ===\n")

print("Top 10 Most Common Objects:\n")

for obj, count in object_counts.most_common(10):
    print(f"{obj}: {count}")

print("\nTotal Unique Classes Detected:")
print(len(object_counts))
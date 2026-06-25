import json

with open("semantic_map.json", "r") as f:
    data = json.load(f)

object_name = input("Enter object name: ")

print(f"\nFrames containing '{object_name}':\n")

found = False

for frame, detections in data.items():

    if object_name in detections:

        print(
            f"{frame} -> {detections[object_name]}"
        )

        found = True
if not found:
    print("Object not found.")
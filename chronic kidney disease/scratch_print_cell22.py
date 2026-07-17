import json

with open("Kidney_Disease.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Cell 22 Source ===")
print("".join(nb["cells"][22]["source"]))

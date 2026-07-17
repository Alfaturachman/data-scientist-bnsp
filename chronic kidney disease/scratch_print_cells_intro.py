import json

with open("Kidney_Disease.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Cells 3 to 17 ===")
for idx in range(3, 18):
    cell = nb["cells"][idx]
    print(f"--- Cell {idx} ({cell['cell_type']}) ---")
    if cell["cell_type"] == "code":
        print("".join(cell["source"])[:300])

import json

with open("Kidney_Disease.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Total cells:", len(nb["cells"]))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))[:100].replace("\n", " ")
    print(f'{i:02d}: {cell["cell_type"]:8s} | {src}')

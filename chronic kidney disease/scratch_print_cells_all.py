import json

with open("Kidney_Disease.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch_nb_source_dump.txt", "w", encoding="utf-8") as f:
    for idx, cell in enumerate(nb["cells"]):
        f.write(f"\n=======================================================\n")
        f.write(f"CELL {idx} ({cell['cell_type']})\n")
        f.write(f"=======================================================\n")
        f.write("".join(cell.get("source", [])))
        f.write("\n")
print("Dumping finished!")

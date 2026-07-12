import json

filepath = r"d:\project sertifikasi\breast cancer\Breast_Cancer.ipynb"

with open(filepath, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print("=== CELL TYPES ===")
print([cell['cell_type'] for cell in notebook['cells'][:15]])

print("\n=== HEADINGS AND MARKDOWN ===")
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        content = "".join(cell['source']).strip()
        first_line = content.split('\n')[0]
        if first_line.startswith('#'):
            print(f"Cell {i+1}: {first_line}")

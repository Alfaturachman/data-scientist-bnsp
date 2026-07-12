import json

filepath = r"d:\project sertifikasi\breast cancer\Breast_Cancer.ipynb"

with open(filepath, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        content = "".join(cell['source']).strip()
        print(f"--- Cell {i+1} (Markdown) ---")
        lines = content.split('\n')
        for line in lines[:10]:
            print(line)
        if len(lines) > 10:
            print("...")
    elif cell['cell_type'] == 'code':
        content = "".join(cell['source']).strip()
        print(f"--- Cell {i+1} (Code) ---")
        lines = content.split('\n')
        for line in lines[:5]:
            print(line)
        if len(lines) > 5:
            print("...")
    print("\n" + "="*80 + "\n")

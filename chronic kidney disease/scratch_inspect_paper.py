filepath = r"d:\project sertifikasi\chronic kidney disease\journal\Machine Learning Techniques in Chronic Kidney Diseases A Comparative Study of Classification Model Performance-2026-07-12_02-43-47.md"

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Let's search for paragraphs containing "34" or "features"
paragraphs = text.split('\n\n')
for idx, p in enumerate(paragraphs):
    if "34" in p or "features" in p:
        # Check if it has any feature names
        if any(kw in p.lower() for kw in ["creatinine", "hemoglobin", "albumin", "urea", "age"]):
            print(f"Paragraph {idx+1}:")
            print(p)
            print("-" * 50)

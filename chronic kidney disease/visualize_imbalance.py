import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style untuk estetika premium
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Membuat figure dengan dua panel (Subplot side-by-side) dengan ukuran lebih tinggi
fig, axes = plt.subplots(1, 2, figsize=(16, 9.5), gridspec_kw={'width_ratios': [1.1, 1]})
fig.patch.set_facecolor('#ffffff')

# Panel Kiri: Grafik Batang Distribusi Target Label (y)
counts = [250, 150]
labels = ['Penyakit Ginjal Kronis\n(CKD / Target = 0)', 'Kontrol Sehat\n(Not CKD / Target = 1)']
percentages = [62.50, 37.50]
colors = ['#ff3b30', '#007aff']  # iOS Red dan iOS Blue (CKD adalah kelas positif klinis)

# Gambar bar chart
bars = axes[0].bar(labels, counts, color=colors, edgecolor='none', width=0.45, 
                   linewidth=0, alpha=0.9, zorder=3)

# Tambahkan garis grid di sumbu y
axes[0].grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# Tambahkan anotasi angka dan persentase di atas batang
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 6,
                 f'{int(height)} Sampel\n({pct:.2f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold',
                 color='#1c1c1e')

# Atur batasan dan visual estetika Panel Kiri
axes[0].set_ylim(0, 280)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_color('#8e8e93')
axes[0].tick_params(bottom=False, left=False, labelleft=True)
axes[0].set_ylabel('Jumlah Sampel', fontsize=12, fontweight='semibold', color='#3a3a3c')
axes[0].set_xticklabels(labels, fontsize=11, fontweight='bold', color='#1c1c1e')
axes[0].set_title('Distribusi Target Label (y)\nDistribusi Kelas Pasien CKD vs Kontrol Sehat', 
                   fontsize=14, fontweight='bold', color='#1c1c1e', pad=25)

# Panel Kanan: Kartu Informasi Dataset & Variabel Klinis (X dan y)
axes[1].axis('off')

# Menggambar latar belakang kartu (grey card)
rect = plt.Rectangle((0.02, 0.01), 0.96, 0.98, fill=True, color='#f2f2f7', 
                     transform=axes[1].transAxes, zorder=1, clip_on=False)
axes[1].add_patch(rect)

# Menambahkan border tipis pada kartu
border = plt.Rectangle((0.02, 0.01), 0.96, 0.98, fill=False, edgecolor='#d1d1d6', linewidth=1.5,
                        transform=axes[1].transAxes, zorder=2, clip_on=False)
axes[1].add_patch(border)

# Judul Kartu Utama
axes[1].text(0.06, 0.94, 'INFORMASI DATASET & VARIABEL KLINIS', fontsize=14, fontweight='bold', color='#1c1c1e', transform=axes[1].transAxes)

# Bagian 1: Sumber & Ukuran Dataset
axes[1].text(0.06, 0.90, 'Sumber & Ukuran Dataset:', fontsize=11, fontweight='bold', color='#007aff', transform=axes[1].transAxes)
dataset_desc = (
    "UCI Chronic Kidney Disease Dataset, terdiri dari 400 sampel klinis\n"
    "dengan 25 kolom (24 fitur prediktor sebagai X, dan 1 target label sebagai y)."
)
axes[1].text(0.06, 0.85, dataset_desc, fontsize=10, color='#3a3a3c', transform=axes[1].transAxes)

# Bagian 2: Fitur Prediktor Utama (X)
axes[1].text(0.06, 0.81, 'Beberapa Fitur Prediktor Kunci (X):', fontsize=11, fontweight='bold', color='#007aff', transform=axes[1].transAxes)

# Daftar Fitur Utama dan Penjelasannya
features = [
    ("hemo", "Hemoglobin - Kadar hemoglobin dalam darah (g/dl)"),
    ("pcv", "Packed Cell Volume - Volume sel darah merah terkemas (%)"),
    ("sg", "Specific Gravity - Berat jenis urin (kategori densitas urin)"),
    ("al", "Albumin - Kandungan protein albumin dalam urin (skala 0-5)"),
    ("sc", "Serum Creatinine - Kadar kreatinin darah (indikator fungsi ginjal)"),
    ("bp", "Blood Pressure - Tekanan darah pasien (mmHg)"),
    ("dm", "Diabetes Mellitus - Riwayat penyakit diabetes (yes/no)"),
    ("htn", "Hypertension - Riwayat penyakit tekanan darah tinggi (yes/no)"),
    ("age", "Age - Usia pasien (tahun)")
]

y_pos = 0.76
for name, desc in features:
    # Menggambar penanda kotak kecil biru
    axes[1].plot(0.07, y_pos, marker='s', markersize=5, color='#007aff', transform=axes[1].transAxes, zorder=3)
    # Nama Fitur
    axes[1].text(0.10, y_pos - 0.008, name, fontsize=10, fontweight='bold', color='#1c1c1e', transform=axes[1].transAxes)
    # Deskripsi Fitur
    axes[1].text(0.10, y_pos - 0.030, desc, fontsize=8.5, color='#48484a', transform=axes[1].transAxes)
    y_pos -= 0.058

# Bagian 3: Target Label (y)
axes[1].text(0.06, 0.19, 'Target Label (y):', fontsize=11, fontweight='bold', color='#007aff', transform=axes[1].transAxes)
target_desc = (
    "classification (ckd -> 0: Positif CKD, notckd -> 1: Kontrol Sehat)\n"
    "• Pasien CKD (Target 0): 250 sampel (62.50%)\n"
    "• Kontrol Sehat (Target 1): 150 sampel (37.50%)"
)
axes[1].text(0.06, 0.12, target_desc, fontsize=10, color='#3a3a3c', transform=axes[1].transAxes)

# Kesimpulan Kelas Target
axes[1].text(0.06, 0.06, '→ Dataset ini relatif seimbang dibandingkan diabetes, namun tetap memiliki\n  ketidakseimbangan kelas ringan yang dikompensasi secara metodologis.', 
             fontsize=9.5, fontweight='bold', color='#ff9500', transform=axes[1].transAxes)

plt.tight_layout()
# Menyimpan gambar dengan resolusi tinggi (300 DPI)
plt.savefig('d:/project sertifikasi/chronic kidney disease/class_imbalance.png', dpi=300, bbox_inches='tight')
print("Grafik berhasil disimpan di d:/project sertifikasi/chronic kidney disease/class_imbalance.png")

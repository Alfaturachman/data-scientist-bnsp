import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style untuk estetika premium
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Data Missing Value per Fitur (Urutkan dari terbesar ke terkecil)
features = [
    'rbc (Red Blood Cell)',
    'rc (Red Blood Cell Count)',
    'wc (White Blood Cell Count)',
    'pot (Potassium)',
    'sod (Sodium)',
    'pcv (Packed Cell Volume)',
    'pc (Pus Cell)',
    'hemo (Hemoglobin)',
    'su (Sugar)',
    'sg (Specific Gravity)',
    'al (Albumin)',
    'bgr (Blood Glucose Random)',
    'bu (Blood Urea)',
    'sc (Serum Creatinine)',
    'bp (Blood Pressure)',
    'age (Age)'
]
counts = [152, 131, 106, 88, 87, 71, 65, 52, 49, 47, 46, 44, 19, 17, 12, 9]
percentages = [38.00, 32.75, 26.50, 22.00, 21.75, 17.75, 16.25, 13.00, 12.25, 11.75, 11.50, 11.00, 4.75, 4.25, 3.00, 2.25]

# Urutkan data secara terbalik agar visualisasi horizontal dimulai dari yang terbesar di atas
data = sorted(zip(features, counts, percentages), key=lambda x: x[2], reverse=False)
sorted_features, sorted_counts, sorted_percentages = zip(*data)

# Membuat figure
fig, ax = plt.subplots(figsize=(12, 7.5))
fig.patch.set_facecolor('#ffffff')

# Palet warna gradient (dari merah menyala untuk persentase tinggi ke biru/abu-abu untuk rendah)
# Kami membuat skema HSL gradient kustom
colors = []
for p in sorted_percentages:
    if p > 30:
        colors.append('#ff3b30')  # Merah (Severity Tinggi)
    elif p > 20:
        colors.append('#ff9500')  # Oranye (Severity Sedang)
    elif p > 10:
        colors.append('#ffcc00')  # Kuning (Severity Ringan)
    else:
        colors.append('#007aff')  # Biru (Minimal)

# Menggambar bar horizontal
bars = ax.barh(sorted_features, sorted_percentages, color=colors, height=0.55, alpha=0.9, zorder=3)

# Tambahkan garis grid vertikal putus-putus
ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)

# Tambahkan label nilai di ujung bar
for bar, count, pct in zip(bars, sorted_counts, sorted_percentages):
    width = bar.get_width()
    ax.text(width + 0.8, bar.get_y() + bar.get_height()/2,
            f'{count} baris ({pct:.2f}%)',
            ha='left', va='center', fontsize=10.5, fontweight='bold', color='#1c1c1e')

# Atur batasan dan label estetika
ax.set_xlim(0, 45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#8e8e93')
ax.spines['bottom'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.set_xlabel('Persentase Missing Values (%)', fontsize=11, fontweight='semibold', color='#3a3a3c', labelpad=10)
ax.set_yticklabels(sorted_features, fontsize=11, fontweight='bold', color='#1c1c1e')
ax.set_title('Identifikasi Missing Values (Data Incompleteness per Fitur Klinis)\nTotal Sampel Dataset = 400 Pasien (UCI Chronic Kidney Disease)', 
             fontsize=14, fontweight='bold', color='#1c1c1e', pad=25)

plt.tight_layout()
# Simpan gambar dengan resolusi 300 DPI
plt.savefig('d:/project sertifikasi/chronic kidney disease/clinical_anomalies.png', dpi=300, bbox_inches='tight')
print("Grafik berhasil disimpan di d:/project sertifikasi/chronic kidney disease/clinical_anomalies.png")

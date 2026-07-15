import matplotlib.pyplot as plt
import seaborn as sns

# Set style untuk estetika premium
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Data Anomali Klinis
features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
counts = [5, 35, 227, 374, 11]
percentages = [0.65, 4.56, 29.56, 48.70, 1.43]

# Urutkan data berdasarkan persentase anomali menurun agar rapi secara visual
data = sorted(zip(features, counts, percentages), key=lambda x: x[2], reverse=True)
sorted_features, sorted_counts, sorted_percentages = zip(*data)

# Membuat figure
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor('#ffffff')

# Palet warna klinis (Merah menyala untuk severity tinggi, Biru/Kuning untuk rendah)
colors = ['#ff3b30', '#ff9500', '#ffcc00', '#5856d6', '#007aff'] 

# Menggambar bar horizontal
bars = ax.barh(sorted_features, sorted_percentages, color=colors, height=0.5, alpha=0.9, zorder=3)

# Tambahkan garis grid vertikal putus-putus
ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)

# Tambahkan label nilai di ujung bar
for bar, count, pct in zip(bars, sorted_counts, sorted_percentages):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{count} baris ({pct:.2f}%)',
            ha='left', va='center', fontsize=11, fontweight='bold', color='#1c1c1e')

# Atur batasan dan label estetika
ax.set_xlim(0, 58)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#8e8e93')
ax.spines['bottom'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.set_xlabel('Persentase Nilai Nol Tidak Logis (%)', fontsize=11, fontweight='semibold', color='#3a3a3c', labelpad=10)
ax.set_yticklabels(sorted_features, fontsize=12, fontweight='bold', color='#1c1c1e')
ax.set_title('Identifikasi Anomali Klinis (Implausible Zeros pada Fitur Vital)\nTotal Sampel Dataset = 768 Pasien', 
             fontsize=14, fontweight='bold', color='#1c1c1e', pad=25)



plt.tight_layout()
# Simpan gambar dengan resolusi 300 DPI
plt.savefig('d:/project sertifikasi/diabetes/clinical_anomalies.png', dpi=300, bbox_inches='tight')
print("Grafik berhasil disimpan di d:/project sertifikasi/diabetes/clinical_anomalies.png")

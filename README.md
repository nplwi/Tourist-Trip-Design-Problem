# Tourist Trip Design Problem (TTDP) — Genetic Algorithm

Proyek Python untuk menyelesaikan **Tourist Trip Design Problem (TTDP)** multi-hari pada studi kasus destinasi wisata di Kota Semarang menggunakan **Genetic Algorithm (GA)** berbasis permutasi, dibandingkan dengan metode pembanding **Greedy Heuristic**.

## Latar Belakang Singkat

**Tourist Trip Design Problem (TTDP)** adalah permasalahan optimasi kombinatorial untuk merancang rencana perjalanan wisata (*itinerary*) multi-hari yang memaksimalkan total kepuasan (*rating* destinasi) tanpa melanggar berbagai batasan praktis wisatawan, seperti:
- Titik keberangkatan dan kepulangan harian tetap di **Hotel (Depot, `id=0`)**.
- **Time Window** (jam buka–tutup setiap destinasi; wisatawan menunggu jika datang lebih awal).
- **Batas Durasi Harian** (`08:00` – `18:00`).
- **Total Anggaran** (`B_max` mencakup tiket masuk dan biaya transportasi per kilometer).
- Setiap destinasi maksimal dikunjungi **1 kali**.

## Formulasi Fitness & Constraint

### Fungsi Fitness
$$\text{Fitness} = \sum_{i \in V} r_i - w_t \cdot T_{\text{travel}}$$

- $r_i$: *rating* destinasi wisata $i$ yang berhasil dikunjungi ($i \in V$).
- $T_{\text{travel}}$: total waktu tempuh perjalanan (menit) menggunakan jarak *Haversine* dengan kecepatan rata-rata $30\text{ km/jam}$.
- $w_t = 0.001$: bobot penalti waktu tempuh sebagai *tie-breaker* untuk efisiensi rute.

### Batasan (*Constraints*)
1. **Depot Harian**: Setiap hari ($D = 2$) dimulai dari Hotel (`id=0`) pukul `08:00` dan harus kembali ke Hotel paling lambat pukul `18:00`.
2. **Jam Buka–Tutup**: Kunjungan pada destinasi $i$ hanya sah apabila waktu selesai kunjungan $\le \text{close\_hour}_i$. Jika tiba sebelum $\text{open\_hour}_i$, wisatawan menunggu hingga destinasi buka.
3. **Anggaran Maksimal**: Total biaya tiket masuk + biaya transportasi ($\text{Rp } 3.000/\text{km}$) selama $D$ hari tidak boleh melebihi $B_{\max} = \text{Rp } 350.000$.
4. **Unik**: Setiap destinasi wisata hanya dapat dikunjungi maksimal 1 kali.

## Struktur Proyek

```text
Tourist-Trip-Design-Problem/
├── data/
│   └── poi_semarang.csv        # Dataset simulasi: 1 Hotel (id=0) + 18 POI Semarang
├── results/
│   ├── best_itinerary.json     # Rute terbaik GA dalam format JSON terstruktur
│   ├── convergence_plot.png    # Grafik konvergensi fitness GA vs Greedy Baseline
│   └── route_plot.png          # Visualisasi koordinat & panah rute per hari
├── src/
│   ├── __init__.py
│   ├── models.py               # Dataclass POI, TripConfig, DayRoute, ItineraryResult
│   ├── decoder.py              # Matriks Haversine & decoder kromosom permutasi
│   ├── ga.py                   # Genetic Algorithm (Tournament, OX, Swap/Inversion, Elitism)
│   └── baseline.py             # Greedy Heuristic (rasio rating / jarak terdekat)
├── main.py                     # Entry point eksperimen, evaluasi, & visualisasi
├── requirements.txt            # Daftar dependensi Python
├── .gitignore
└── README.md
```

## Cara Instalasi & Menjalankan

1. **Instalasi dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Menjalankan eksperimen**:
   ```bash
   python main.py
   ```
   Skrip ini akan menjalankan **Greedy Baseline** dan **10 *independent runs* Genetic Algorithm** (`seed=42..51`, `population_size=80`, `generations=150`), mencetak ringkasan perbandingan serta jadwal lengkap rute terbaik di terminal, dan menyimpan hasil ke folder `results/`.

## Ringkasan Hasil Uji

Hasil eksekusi aktual `main.py` pada dataset `data/poi_semarang.csv` (18 POI, 2 hari, `B_max = Rp 350.000`, `08:00–18:00`):

| Metode | Fitness | Total Rating | POI Dikunjungi | Waktu Tempuh | Total Biaya | Validasi Kendala |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy Baseline** | `67.8807` | `68.0` | `15` | `119.3 mnt` | `Rp 328,956` | Valid (`True`) |
| **GA Best (`seed=48`)** | **`67.8980`** | **`68.0`** | **`15`** | **`102.0 mnt`** | **`Rp 303,032`** | **Valid (`True`)** |

- **Rata-rata Fitness GA (10 runs)**: `65.9276 ± 1.9658`
- **Rata-rata Runtime GA (10 runs)**: `1.6375 detik`
- **Peningkatan GA Best vs Greedy Baseline**: `+0.03%` pada fitness (dengan penghematan waktu tempuh **17.3 menit** dan penghematan biaya **Rp 25.924** untuk jumlah POI dan total rating yang sama).

### Rute Terbaik GA (`seed=48`)
- **Hari 1** (`08:00` – `17:41`, 8 POI, Rating `36.9`, Jarak `35.71 km`, Biaya `Rp 172,127`):
  `Hotel` → `Pura Agung Giri Natha` → `Pagoda Avalokitesvara` → `Brown Canyon Semarang` → `Masjid Agung Jawa Tengah` → `Semarang Contemporary Art Gallery` → `Gereja Blenduk` → `Kota Lama Semarang` → `Tugu Muda Semarang` → `Hotel`
- **Hari 2** (`08:00` – `17:46`, 7 POI, Rating `31.1`, Jarak `15.30 km`, Biaya `Rp 130,906`):
  `Hotel` → `Kampung Pelangi` → `Klenteng Sam Poo Kong` → `Museum Ronggowarsito` → `Grand Maerakaca` → `Pantai Marina` → `Lawang Sewu` → `Simpang Lima Park` → `Hotel`

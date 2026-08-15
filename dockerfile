# 1. Gunakan base image Python (sesuaikan dengan versi Python yang Anda pakai, misal 3.9 atau 3.10)
FROM python:3.12.13

# 2. Set working directory di dalam container
WORKDIR /app

# 3. Copy file requirements terlebih dahulu (ini trik agar Docker melakukan cache pada layer instalasi)
COPY config/requirements.txt .

# 4. Install semua dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy seluruh file dan folder project dari komputer Anda ke dalam container
COPY . .

# 6. Tentukan COMMAND SPESIFIK yang dieksekusi saat container berjalan.
# Contoh: Menjalankan pipeline utama. Sesuaikan path-nya jika perlu memanggil config.
CMD ["./run_pipeline.sh"]
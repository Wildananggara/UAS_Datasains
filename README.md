# Streamlit - Prediksi Flow Lalu Lintas (Paris)

## Struktur
- `app.py` : aplikasi Streamlit
- `requirements.txt` : dependency
- (opsional) `paris.csv` : dataset (kalau mau dibundle di repo)

## Jalankan lokal
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Aplikasi akan:
1) memakai file yang kamu upload via UI, atau
2) jika `paris.csv` ada di folder yang sama dengan `app.py`, otomatis dipakai.

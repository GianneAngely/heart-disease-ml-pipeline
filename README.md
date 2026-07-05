# Eksperimen_SML_Gianne-Angely

Repository ini berisi eksperimen machine learning untuk dataset Heart Disease UCI sebagai bagian dari submission kelas Membangun Sistem Machine Learning.

## Struktur Folder

```
Eksperimen_SML_Gianne-Angely
├── .github/workflows/
│   └── preprocessing.yml
├── heart_disease_raw/
│   └── heart.csv
├── preprocessing/
│   ├── Eksperimen_Gianne-Angely.ipynb
│   ├── automate_Gianne-Angely.py
│   └── heart_disease_preprocessing/
│       └── heart_preprocessed.csv
```

## Dataset

Dataset yang digunakan adalah **Heart Disease UCI** dari Kaggle, dengan 303 sampel dan 14 fitur.

## Cara Menjalankan Preprocessing Otomatis

```bash
pip install -r preprocessing/requirements.txt
python preprocessing/automate_Gianne-Angely.py
```

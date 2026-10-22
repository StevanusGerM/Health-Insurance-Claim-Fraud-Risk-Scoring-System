# Health Insurance Claim Fraud Risk Scoring System

Sistem sokongan keputusan berasaskan pembelajaran mesin dan statistik yang direka khusus untuk menangani isu penipuan tuntutan insurans kesihatan di bawah kekangan kapasiti audit maksimum sebanyak 5%.

## **A. Gambaran Keseluruhan Kes (Case Overview)**
Sistem kesihatan nasional sering berdepan dengan cabaran kerugian akibat penipuan yang dianggarkan antara 3% hingga 10% daripada jumlah perbelanjaan tahunan. Memandangkan semakan atau audit manual bagi setiap tuntutan adalah tidak munasabah dari segi operasi dan kewangan, had kapasiti audit ditetapkan sebanyak 5%. 

Projek ini dibangunkan untuk memastikan tuntutan yang dipilih dalam had kapasiti audit tersebut benar-benar kes penipuan dengan memfokuskan kepada metrik **NormalizedRecall@5%**. Sistem ini menggabungkan:
* Ciri-ciri klinikal dan pentadbiran.
* Model hibrid **LightGBM** dan **CatBoost** yang disahkan melalui 5-fold cross-validation.
* Penentukuran kebarangkalian menggunakan **Isotonic Regression** bagi memastikan skor risiko mudah diinterpretasi oleh verifikator manusia.

## **B. Gambaran Keseluruhan Data (Data Overview)**
* **Train Data (`train.csv`)**: Mengandungi 160,174 baris dan 53 kolum (51 ciri, 1 ID tuntutan, dan 1 kolum sasaran/label) yang bersih tanpa nilai hilang (*missing values*).
* **Test/Inference Data (`test.csv`)**: Mengandungi 40,043 baris dan 52 kolum (tanpa kolum label).
* **Kategori Pembolehubah Utama**:
  * Informasi Geografi / Pentadbiran (`kdkc`, `dati2`)
  * Fasiliti Kesihatan (`typeppk`)
  * Demografi Pesakit (`jkpst`, `umur`)
  * Jenis Perkhidmatan & Tempoh Inap (`jnspelsep`, `los`)
  * Diagnosis Utama & Sekunder ICD-10 (`diagprimer`, `dx2_*`)
  * Kod Prosedur (`proc*`)

## **C. Penjanaan Ciri (Feature Engineering)**
Skrip pemprosesan membina pelbagai ciri tambahan (*derived features*) untuk meningkatkan prestasi model, antaranya:
* Pengiraan jumlah diagnosis sekunder dan prosedur (`has_secondary_dx`, `total_secondary_dx`, `has_procedure`, `total_procedure`)
* Pengkategorian umur dan tempoh inap (`age_group`, `los_bucket`)
* Interaksi tahap keterukan (*severity level*) dengan diagnosis dan tempoh inap
* Analisis saiz duplikasi logik (`dup_group_size`, `is_duplicate`) untuk mengesan corak tuntutan berulang yang mencurigakan

## **D. Keperluan Pustaka (Dependencies)**
Pastikan pustaka Python berikut dipasang sebelum menjalankan kod:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import *
from catboost import CatBoostClassifier
import lightgbm as lgb

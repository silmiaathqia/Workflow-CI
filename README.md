# ⚙️ Worker Productivity MLflow CI/CD Pipeline
### Kriteria 3 — Workflow CI dengan GitHub Actions + MLflow Project

<div align="center">

![CI/CD](https://github.com/silmiaathqia/Workflow-CI/actions/workflows/ci-mlflow.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12.7-blue?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-orange?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

<br/>

**Kriteria 3 — Proyek Akhir Kelas Membangun Sistem Machine Learning — Dicoding**

*oleh Silmi Azdkiatul Athqia (silmiathqia)*

</div>

> Pipeline CI/CD otomatis untuk re-training model klasifikasi produktivitas pekerja menggunakan **MLflow Project** dan **GitHub Actions**, dengan integrasi penuh ke **DagsHub** dan **Docker Hub**.

---

## 🗺️ Navigasi Proyek MSML

> Repo ini adalah bagian dari proyek akhir MSML yang terdiri dari 3 repo yang saling terhubung:

```
📦 Eksperimen_SML (K1)          ⚙️ Workflow-CI (K3)          📊 SMSML Submission
─────────────────────           ────────────────────           ──────────────────
EDA + Preprocessing      →      CI/CD + mlflow run .    →      Submission Lengkap
automate_silmi.py               GitHub Actions                  K1 + K2 + K3 + K4
GitHub Actions (K1)             MLflow Project                  Monitoring Grafana
         │                              │                              │
         └──────────────────────────────┴──────────────────────────────┘
                              Dataset yang sama
                    remote_worker_productivity_preprocessing
```

| Repo | Kriteria | Deskripsi |
|---|:---:|---|
| [Eksperimen_SML_Silmi-Azdkiatul-Athqia](https://github.com/silmiaathqia/Eksperimen_SML_Silmi-Azdkiatul-Athqia) | K1 | EDA, preprocessing, automate script |
| **Workflow-CI** ← *kamu di sini* | K3 | CI/CD pipeline, MLflow Project, Docker |
| [SMSML_Silmi-Azdkiatul-Athqia](https://github.com/silmiaathqia/SMSML_Silmi-Azdkiatul-Athqia) | K1-K4 | Submission utama lengkap |

---

## 📋 Deskripsi

Repository ini mengimplementasikan **Workflow CI** menggunakan GitHub Actions dan MLflow Project untuk re-training otomatis model MLP Classifier klasifikasi produktivitas pekerja remote. Setiap kali trigger dipantik, workflow akan:

1. Setup environment Conda
2. Validasi dataset
3. Jalankan `mlflow run .` sebagai mekanisme training utama
4. Simpan artefak ke GitHub
5. Build dan push Docker image ke Docker Hub

---

## 📂 Struktur Repository

```
Workflow-CI/
├── 📁 .github/workflows/
│   └── ⚙️ ci-mlflow.yml          # Main CI/CD workflow
├── 📁 MLProject/
│   ├── 🐍 modelling.py            # Script training utama
│   ├── ⚙️ conda.yaml              # Environment dependencies
│   ├── 📄 MLproject               # MLflow project config
│   └── 📁 processed_data/         # Dataset preprocessed
│       ├── data_train.csv
│       ├── data_validation.csv
│       └── data_test.csv
└── 📄 README.md
```

---

## 🧠 Model

**MLP Classifier (Scikit-Learn)**

| Parameter | Nilai |
|---|---|
| Hidden Layers | (128, 64, 32) |
| Activation | ReLU |
| Solver | Adam |
| Alpha (L2) | 0.001 |
| Early Stopping | ✅ Enabled |
| Target Classes | High, Low, Medium |

---

## 🔄 CI/CD Pipeline

### Trigger

```yaml
on:
  push:         # Push ke main/master
  pull_request: # PR ke main/master
  workflow_dispatch: # Manual trigger
```

### Alur Pipeline

```
Checkout Repository
    │
    ▼
Setup Conda (Python 3.12.7)
    │
    ▼
Create MLProject Structure
    │
    ▼
Install Dependencies (conda.yaml)
    │
    ▼
Validate Data Files
    │
    ▼
mlflow run . --env-manager=conda   ← mekanisme training utama
    │
    ▼
Build Docker Image (Python 3.12.7)
    │
    ▼
Push ke Docker Hub
    │
    ▼
Upload Artifacts ke GitHub
    │
    ▼
Create GitHub Release ✅
```

---

## ⚙️ GitHub Secrets

| Secret | Deskripsi |
|---|---|
| `MLFLOW_TRACKING_URI` | URL DagsHub MLflow tracking |
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_USER_TOKEN` | Token autentikasi DagsHub |
| `DOCKER_USERNAME` | Username Docker Hub |
| `DOCKER_PASSWORD` | Password Docker Hub |
| `DOCKER_REPO` | Nama repo Docker Hub |

---

## 🐳 Docker

```bash
# Pull image
docker pull silmiathqia/worker-productivity-mlp:latest

# Run container
docker run -p 8080:8080 silmiathqia/worker-productivity-mlp:latest
```

---

## 📦 Dependencies

| Library | Versi |
|---|---|
| Python | 3.12.7 |
| MLflow | 2.19.0 |
| Scikit-Learn | 1.5.2 |
| Pandas | 2.3.0 |
| NumPy | 1.26.4 |
| DagsHub | 0.5.10 |

---

## 🔗 Links

| Resource | Link |
|---|---|
| 📈 MLflow Tracking | [DagsHub](https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow) |
| 🐳 Docker Image | [Docker Hub](https://hub.docker.com/r/silmiathqia/worker-productivity-mlp) |
| 📦 Repo K1 | [Eksperimen_SML](https://github.com/silmiaathqia/Eksperimen_SML_Silmi-Azdkiatul-Athqia) |
| 📊 Submission Utama | [SMSML](https://github.com/silmiaathqia/SMSML_Silmi-Azdkiatul-Athqia) |

---

## 🎓 Sertifikat

<div align="center">

> 🏅 **Membangun Sistem Machine Learning** — Dicoding Indonesia
>
> Diperoleh oleh **Silmi Azdkiatul Athqia**

[![Lihat & Verifikasi Sertifikat](https://img.shields.io/badge/🎓%20Lihat%20Sertifikat-Dicoding-06b6d4?style=for-the-badge)](https://www.dicoding.com/certificates/JMZVOJ9O3XN9)

</div>

---

## 👩‍💻 Author

<div align="center">

**Silmi Azdkiatul Athqia**

[![Dicoding](https://img.shields.io/badge/Dicoding-silmiathqia-blue?style=flat-square)](https://www.dicoding.com/users/silmiathqia)
[![GitHub](https://img.shields.io/badge/GitHub-silmiaathqia-black?style=flat-square&logo=github)](https://github.com/silmiaathqia)

🎓 Laskar AI 2025 Cohort — Mahasiswa & Fresh Graduate

</div>

---

<div align="center">
<i>Kriteria 3 — Membangun Sistem Machine Learning (MSML) — Dicoding 2025</i>
<br/>
<sub>Made with ❤️ by Silmi Azdkiatul Athqia</sub>
</div>
